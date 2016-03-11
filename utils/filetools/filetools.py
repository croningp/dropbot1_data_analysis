import os
import fnmatch


def ensure_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def is_subdir(path, directory):
    path = os.path.realpath(path)
    directory = os.path.realpath(directory)
    relative = os.path.relpath(path, directory)
    if relative.startswith(os.pardir):
        return False
    else:
        return True


def list_files(path='.', patterns=['*'], min_depth=0, max_depth=float('inf')):
    if type(patterns) == str:
        patterns = [patterns]
    found_files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            filedir = os.path.abspath(dirpath)
            filepath = os.path.join(filedir, filename)
            depth = filepath[len(os.path.abspath(path)) + len(os.path.sep):].count(os.path.sep)
            if min_depth <= depth <= max_depth:
                for pattern in patterns:
                    if fnmatch.fnmatch(filename, pattern):
                        found_files.append(filepath)
    return found_files


def list_folders(path='.'):
    paths = [os.path.abspath(os.path.normpath(os.path.join(path, x))) for x in os.listdir(path)]
    return filter(lambda x: os.path.isdir(x), paths)


class File(object):
    def __init__(self, filepath, refpath='.'):
        self.refpath = os.path.abspath(os.path.normpath(refpath))
        if os.path.isabs(filepath):
            self.path = os.path.normpath(filepath)
        else:
            self.path = os.path.join(self.refpath, filepath)

    @property
    def abspathfromref(self):
        return os.path.join(os.sep, self.relpath)

    @property
    def absdirnamefromref(self):
        return os.path.join(os.sep, self.reldirname)

    @property
    def abspath(self):
        return os.path.abspath(self.path)

    @property
    def relpath(self):
        return os.path.relpath(self.path, self.refpath)

    @property
    def reldirname(self):
        return os.path.relpath(self.dirname, self.refpath)

    @property
    def exists(self):
        return os.path.exists(self.path)

    @property
    def filebasename(self):
        return os.path.splitext(self.filename)[0]

    @property
    def extension(self):
        return os.path.splitext(self.filename)[1][1:]

    @property
    def filename(self):
        return os.path.basename(self.path)

    @property
    def dirname(self):
        return os.path.dirname(self.path)

    @property
    def modified(self):
        return os.path.getmtime(self.path)

    @property
    def created(self):
        return os.path.getctime(self.path)

    def change_filename(self, newfilename):
        self.path = os.path.join(self.dirname, newfilename)

    def change_filebasename(self, newbasename):
        self.path = os.path.join(self.dirname, newbasename + os.extsep + self.extension)

    def change_ext(self, newext):
        self.path = os.path.join(self.dirname, self.filebasename + os.extsep + newext)

    def change_refpath(self, newrefpath):
        self.path = os.path.normpath(os.path.join(newrefpath, self.relpath))
        self.refpath = os.path.normpath(newrefpath)

    def is_older(self, file):
        return self.modified < file.modified

    def duplicate(self, newrefpath=None):
        file_copy = File(self.path, self.refpath)
        if newrefpath:
            file_copy.change_refpath(newrefpath)
        return file_copy

    def read(self):
        with open(self.path) as f:
            data = f.read()
        return data

    def write(self, data):
        ensure_dir(self.dirname)
        with open(self.path, 'w') as f:
            f.write(data)

    def append(self, data):
        ensure_dir(self.dirname)
        with open(self.path, 'a') as f:
            f.write(data)
