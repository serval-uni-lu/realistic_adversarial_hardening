mkdir package
mkdir -p package/src/attacks
cp -r config package/
cp src/* package/src
cp -r src/attacks/coeva2 package/src/attacks/
cp -r src/utils package/src/utils
rm -r package/src/attacks/coeva2/__pycache__
rm -r package/src/utils/__pycache__
cp requirements-min.txt package/

zip -r package.zip package
rm -r package
