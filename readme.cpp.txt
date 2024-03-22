1. pip install pybind11


2. find installation and run pybind11-config using --includes:

leomegliola@leos-air bin % ./pybind11-config --includes
-I/Library/Python/3.9/include -I/Users/leomegliola/Library/Python/3.9/lib/python/site-packages/pybind11/include


3. find the python dev library:

find /usr /System /Library -name "libpython3.9.dylib" -print 2>/dev/null



4. compile is: g++ [includes] [source] [target] [options] [libs]

g++ 
-I/Users/leomegliola/Library/Python/3.9/lib/python/site-packages/pybind11/include 
-I/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Headers 
pybind_test.cpp -o 
test_module.so 
-std=c++11 
-fPIC 
-shared 
-L/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib 
-lpython3.9

g++ -I/Users/leomegliola/Library/Python/3.9/lib/python/site-packages/pybind11/include -I/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Headers pybind_test.cpp -o test_module.so -std=c++11 -fPIC -shared -L/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib -lpython3.9
