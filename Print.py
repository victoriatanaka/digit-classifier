class Print:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def header(string):
    print(Print.HEADER + str(string) + Print.ENDC)

def okBlue(string):
    print(Print.OKBLUE + str(string) + Print.ENDC)

def okGreen(string):
    print(Print.OKGREEN + str(string) + Print.ENDC)

def warning(string):
    print(Print.WARNING + str(string) + Print.ENDC)

def fail(string):
    print(Print.FAIL + str(string) + Print.ENDC)

def bold(string):
    print(Print.BOLD + str(string) + Print.ENDC)

def underline(string):
    print(Print.UNDERLINE + str(string) + Print.ENDC)


