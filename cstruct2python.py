

file1 = open('Input.h', 'r')
startClass = False

cTypesToCTypesMap = {
    "unsigned char":"c_ubyte",
    "unsigned long":"c_int",    
    "char":"c_char",    
    "long":"c_int",    
    "float":"c_float",    
    "int":"c_int",    
    "bool":"c_bool",    
    "double":"c_double",    
}

print("import ctypes\nfrom ctypes import *\n\n")

while True:
    # Get next line from file
    line = file1.readline()
    if not line:
        break
    line = line.strip()
    if line.startswith("struct"):
        startClass = True
        className = line.split(" ")[1]
        print("class",className,"(Structure):\n\t_pack_ = 4\n\t_fields_ = [")
    elif line.startswith("//"):
        # print("#",line)
        pass
    elif "};" in line and startClass:
        startClass = False
        print("\t]")
    elif ";" in line and startClass:
        line = line.split(";")[0].strip()
        parts = line.split(" ")
        variableName = ""
        typeName = ""
        if len(parts) == 2:
            variableName = parts[1]
            typeName = cTypesToCTypesMap[parts[0]]
        else:
            variableName = parts[2]
            typeName = cTypesToCTypesMap[parts[0]+" "+parts[1]]

        if "[" in variableName:
            parts = variableName.split("[")
            variableName = parts[0]
            typeName = typeName+" * "+parts[1][0:-1]
            if len(parts)>2:
                typeName = typeName+" * "+parts[2][0:-1]

        print("\t\t('"+variableName+"', "+typeName+"),")
