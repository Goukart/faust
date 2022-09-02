# !/bin/bash

# Requires images in "render_out"
# RUN
# ./pipe.sh imageRegex="./mm_out/.*[0]-.*.png" resolution=348 output="render" skipTo=0

# necesarry parameters:
# expression to select images -> will be used in micmac as well
for P; do
  eval $P
done

# This should manage all and orchestrate it
# when one script aborts no other should be executed
# same in micmac, if one step fails dont run following ones

if [[ $path = "" ]]; then
  echo "MicMac working directory [path] is required."
  exit 1
fi

# things to consider:
# mm_out and those directories should be created by script

# Inject the rendered images with real meta data
## echo "Do incection of meta data"
## conda activate faust
## python3 inject.py $imageRegex
## conda deactivate
## echo "Injection finished"

# Prepare directory for micmac and run micmac
# rm -r $(find mm_out/ -mindepth 1 -maxdepth 1 ! -name "micmac.sh")
## mv injection_out/* mm_out/
# Possibly change later?
cd $path
./micmac.sh path=$path imageRegex=$imageRegex resolution=$resolution output=$output skipTo=$skipTo

# copy to path and create if it does not exist
# mkdir -p /foo/bar && cp myfile "$_"
# mkdir -p "$d" && cp file "$d"
# rsync -a myfile /foo/bar/
