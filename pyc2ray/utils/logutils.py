def printlog(s, filename, quiet=False, end="\n"):
    """Write log and print to screen

    Parameters
    ----------
    s : str
        String to write to log
    filename : str
        Name of the logfile to append text to
    quiet : bool
        Don't write to stdout. Default is False
    """

    with open(filename, "a") as f:
        f.write(s + end)
    if not quiet:
        print(s, end=end)
