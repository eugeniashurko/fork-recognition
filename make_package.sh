mkdir dist
mkdir dist/library
mkdir dist/bin

cp quick_class.py dist/
cp quick_learn.py dist/
cp class.py dist/
cp learn.py dist/
cp distance.py dist/
cp X_quick dist/
cp y_quick dist/
cp script* dist/
cp install_dependencies.sh dist/
cp pip_req.txt dist/

cp library/*.py dist/library/
cp bin/* dist/bin/

#cp database dist/database