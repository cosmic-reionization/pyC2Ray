"""Giri2024 per-halo mass moments and their conversion to source rates.

W=sum M*(M/Mpivot)**alpha_star, in Msun; no f_b or fstar0 included.
Gradual G=sum_L W_i*clip((M_i-Mmin)/(Mmax-Mmin),0,1).
Optional E=sum W_i*min(1,fesc0*(M_i/Mpivot)**alpha_esc).
For constant escape, E=fesc0*W. A corresponding gradual E is also stored. Never estimate E from a cell's total mass.
The catalogue redshift sets H(z); rates are held fixed across RT substeps.
"""
import json
from pathlib import Path
import numpy as np

MSUN = 1.988409870698051e33
MP = 1.67262192369e-24
YEAR = 3.15576e7


def parameters(config):
    p = config['Giri']
    keys = ['fstar0','alpha_star','pivot_msun','alpha_h','Nion','fesc0','alpha_esc']
    d = {k:float(p[k]) for k in keys}
    if not all(np.isfinite(list(d.values()))) or min(d['pivot_msun'],d['alpha_h'],d['Nion']) <= 0:
        raise ValueError('Invalid Giri parameters.')
    if not 0 <= d['fstar0'] <= 1 or not 0 <= d['fesc0'] <= 1:
        raise ValueError('fstar0 and fesc0 must be in [0,1].')
    d['gradual_mass_min_msun'] = float(p.get('gradual_mass_min_msun', 1e8))
    d['gradual_mass_max_msun'] = float(p.get('gradual_mass_max_msun', 1e9))
    if not np.isfinite(d['gradual_mass_min_msun']) or not np.isfinite(d['gradual_mass_max_msun']) or not 0 < d['gradual_mass_min_msun'] < d['gradual_mass_max_msun']:
        raise ValueError('Invalid gradual suppression mass bounds.')
    return d


def gradual_retention(mass_msun, p):
    """Fraction retained, linear in MASS (not log mass)."""
    return np.clip((np.asarray(mass_msun) - p['gradual_mass_min_msun']) /
                   (p['gradual_mass_max_msun'] - p['gradual_mass_min_msun']), 0., 1.)


def halo_weights(mass_msun, p):
    m=np.asarray(mass_msun,dtype=float)
    if np.any(~np.isfinite(m)) or np.any(m<=0):
        raise ValueError('Invalid halo masses.')
    w=m*(m/p['pivot_msun'])**p['alpha_star']
    if np.any(~np.isfinite(w)) or np.any(p['fstar0']*w/m>1):
        raise ValueError('Giri efficiency exceeds one or is nonfinite; extrapolation is not supported.')
    esc=np.minimum(1.,p['fesc0']*(m/p['pivot_msun'])**p['alpha_esc'])
    return w, w*esc


def read_catalogue(file, config, ngrid, n_box, h, omega_m, boxsize):
    p=parameters(config)
    meta=json.loads(Path(str(file)+'.meta.json').read_text())
    if meta['format'] not in ('giri_mass_moments_v1', 'giri_mass_moments_v2'):
        raise ValueError('Unsupported Giri catalogue metadata.')
    for key,expected in [('ngrid',ngrid),('n_box',n_box),('h',h),('omega_m',omega_m),('boxsize_Mpc_h',boxsize)]:
        if not np.isclose(meta[key],expected,rtol=1e-8,atol=0):
            raise ValueError('Catalogue/simulation mismatch: '+key)
    for key in ('alpha_star','pivot_msun'):
        if not np.isclose(meta['Giri'][key],p[key],rtol=1e-12,atol=0):
            raise ValueError('Regenerate source moments after changing '+key)
    if p['fstar0']*meta['max_shape_factor']>1:
        raise ValueError('New fstar0 would exceed unity for a catalogue halo.')
    with open(file) as f:
        count=int(f.readline().strip())
        data=np.loadtxt(f,ndmin=2) if count else np.empty((0,len(meta['columns'])))
    if data.shape!=(count,len(meta['columns'])) or data.shape[1] not in ((8,11) if meta['format']=='giri_mass_moments_v2' else (7,9)):
        raise ValueError('Incorrect Giri source row/column count.')
    if np.any(~np.isfinite(data)) or np.any(data[:,3:]<0):
        raise ValueError('Invalid source entries.')
    pos=data[:,:3]-1
    if np.any(pos!=np.floor(pos)) or np.any(pos<0) or np.any(pos>=ngrid):
        raise ValueError('Invalid catalogue source positions.')
    w=data[:,5:7]
    v2 = meta['format']=='giri_mass_moments_v2'
    has_escape = data.shape[1] == (11 if v2 else 9)
    if has_escape:
        for key in ('fesc0','alpha_esc'):
            if not np.isclose(meta['Giri'][key],p[key],rtol=1e-12,atol=0):
                raise ValueError('Regenerate escape moments after changing '+key)
        e=data[:,8:10] if v2 else data[:,7:9]
    else:
        if p['alpha_esc']!=0:
            raise ValueError('Mass-dependent escape requires --escape-moments (11 columns for v2).')
        e=w*p['fesc0']
    wg = data[:,7] if v2 else None
    eg = (data[:,10] if has_escape else wg*p['fesc0']) if v2 else None
    if config['Giri'].get('lmach_suppression') == 'gradual':
        if not v2:
            raise ValueError('Gradual mode requires regenerated v2 catalogues (8 or 11 columns).')
        for key in ('gradual_mass_min_msun','gradual_mass_max_msun'):
            if not np.isclose(meta['Giri'][key],p[key],rtol=1e-12,atol=0):
                raise ValueError('Regenerate gradual moments after changing '+key)
        if np.any(wg > w[:,1]*(1+1e-12)) or np.any(eg > e[:,1]*(1+1e-12)):
            raise ValueError('Gradual moments exceed unsuppressed moments.')
    return pos.astype(np.intp).T,data[:,3:5],w,e,wg,eg,meta,p


def budgets(w,e,p,z,H_s,fb,lw_ratios,lw_escape,suppression):
    """w/e: (Ncell,2); suppression: LMACH factor (Ncell,)."""
    if not np.isfinite(z) or z<=-1 or not np.isfinite(H_s) or H_s<=0 or not 0<fb<=1:
        raise ValueError('Invalid source cosmology.')
    scale=np.column_stack([np.ones(len(w)),suppression])
    if np.any(~np.isfinite(scale)) or np.any(scale<0) or np.any(scale>1):
        raise ValueError('Suppression factors must be in [0,1].')
    lw_ratios=np.asarray(lw_ratios);lw_escape=np.asarray(lw_escape)
    if np.any(~np.isfinite(lw_ratios)) or np.any(lw_ratios<0) or np.any(~np.isfinite(lw_escape)) or np.any(lw_escape<0) or np.any(lw_escape>1):
        raise ValueError('Invalid LW spectrum ratio or escape fraction.')
    rate=p['fstar0']*fb*p['alpha_h']*(1+z)*H_s
    sfr=rate*w*scale  # Msun/s
    q_intrinsic=p['Nion']*MSUN/MP*sfr
    q_escape=p['Nion']*MSUN/MP*rate*e*scale
    lw=q_intrinsic*lw_ratios*lw_escape
    return sfr,q_intrinsic,q_escape,lw
