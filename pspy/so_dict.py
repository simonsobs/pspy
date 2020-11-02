"""
@brief: python routines for reading a dictionnary file
"""


def ask_for(key):
    s = input("so_dict: enter value for '%s': " % key)
    try:
        val = eval(s)
    except NameError:
        # allow people to enter unquoted strings
        val = s
    return val


class so_dict(dict):
    def __init__(self, ask=False):
        """
        @param ask if the dict doesn't have an entry for a key, ask for the associated value and assign
        """
        dict.__init__(self)
        self.ask = ask

    def __getitem__(self, key):
        if key not in self:
            if self.ask:
                print("so_dict: parameter '%s' not found" % key)
                val = ask_for(key)
                print("so_dict: setting '%s' = %s" % (key, repr(val)))
                dict.__setitem__(self, key, val)
            else:
                raise ValueError("Missing '{}' key value in dictionary".format(key))
        return dict.__getitem__(self, key)

    def get(self, key, default=None, raise_error=False):
        res = super().get(key, default)
        if res is None and raise_error:
            raise ValueError("Missing '{}' key value in dictionary".format(key))
        return res

    def read_from_file(self, filename):
        f = open(filename)
        old = ""
        for line in f:
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue
            s = line.split("#")
            line = s[0]
            s = line.split("\\")
            if len(s) > 1:
                old = old.join(s[0])
                continue
            else:
                line = old.join(s[0])
                old = ""
            for i in range(len(line)):
                if line[i] != " ":
                    line = line[i:]
                    break
            exec(line, globals())
            s = line.split("=")
            if len(s) != 2:
                print("Error parsing line:")
                print(line)
                continue
            key = s[0].strip()
            val = eval(s[1].strip())  # XXX:make safer
            self[key] = val
        f.close()

    readFromFile = read_from_file

    def write_to_file(self, filename, mode="w"):
        f = open(filename, mode)
        keys = self.keys()
        keys.sort()
        for key in keys:
            f.write("%s = %s\n" % (key, repr(self[key])))
        f.close()

    writeToFile = write_to_file
