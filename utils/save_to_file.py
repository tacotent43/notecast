def save_to_file(
    thing: str | dict,
    path: str,
):
    if isinstance(thing, dict):
        pass
    elif isinstance(thing, str):
        with open(path, "w") as outputfile:
            outputfile.write(thing)
