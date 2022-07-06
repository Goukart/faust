# !/bin/bash

# Parse arguments as praram=value pairs, if additionals are providede doesent
# matter it they aren't used
# tho I'd have to check if they change any I already use?
# clean -> remove all; as a command
# cleanup -> leave relevant files
# man or help menu
# autocomplete like fish and zsh
# save last used name to better clean up?

# https://stackoverflow.com/questions/192319/how-do-i-know-the-script-file-name-in-a-bash-script
# > get the script name:
# me=$(basename -- "$0") or much more efficiently at the expense of readability, me=${0##*/}
me=${0##*/}
# echo inside $me

for P; do
  eval $P
done

# Check expression validity
# show what images are selected


# implement dry run for debug
# step=# only run that step
# micmac is kinda shitty, seems to use first folder in directory:
# *I have a folder "aussetzen" to move images I dont want to use right now*
# Image 1.jpg, with ori : Ori-1110_Hand/Orientation-1.jpg.xml
# Image 2.jpg, with ori : Ori-1110_Hand/Orientation-2.jpg.xml
# Image 3.jpg, with ori : Ori-1110_Hand/Orientation-3.jpg.xml
# For required file [./Ori-1110_Hand/Orientation-aussetzen/4.jpg.xml]
# ------------------------------------------------------------
# |   Sorry, the following FATAL ERROR happened
# |
# |    Cannot open
# |
# ------------------------------------------------------------

#while getopts i:r:o:s: option
#do
#  case $option in
#    i)
#      imageRegex="${OPTARG}"
#      echo "imageRegex was set to ${OPTARG}"
#      ;;
#    r)
#      resolution=${OPTARG}
#      ;;
#    o)
#      output="${OPTARG}"
#      ;;
#    s)
#      skipTo=${OPTARG}
#      ;;
#    \?)
#      echo "Invalid option -$OPTARG" >&2
#      exit 1
#      ;;
#    :)
#      echo "Option -$OPTARG requires an argument." >&2
#      exit 1
#    ;;
#  esac
#done

# ToDo explain how it works
# create help screen
# -c to delete everything before restart -> clean build
# --clean alone like --help to delete all but this

if [[ $imageRegex = "" ]]; then
  echo "Image pattern [imageRegex] is required."
  exit 1
fi
if [[ $resolution = "" ]]; then
  echo "Target resolution [resolution] is required."
  exit 1
fi
if [[ $output = "" ]]; then
  echo "Named result [output] is required."
  exit 1
fi


#mask=$4.jpg
mesh="Mesh-${output}-${resolution}.ply"
final="Mesh-${output}-${resolution}_textured.ply"
c3cdMode="BigMac"

function section () {
  echo "##################################################################"
  echo "#"
  echo "#             "$1
  echo "#"
  echo "##################################################################"
}
# Cleanup command (only png and jpg survive the purge):
# rm  -r $(find . -name "*" ! -name "micmac.sh")
# rm -r $(find . -maxdepth 1 -name "*" ! \( -name "micmac.sh" -o -iname \*.png -o -iname \*.jpg \))

#function micmac(){
  echo "Selecting all images [$imageRegex] with a [$resolution px] width."
  #echo "Mask file [$mask]"
  echo "Output file: [$final]"
  sleep 2 &&
  START_TIME=$(date +%s) &&

  if [[ $skipTo > 1 ]]; then
    echo "Skipping: Find Tie-Points"
  else
    section "First Step: Find Tie-Points"
    mm3d Tapioca All $imageRegex $resolution
  fi

  if [[ $skipTo > 2 ]]; then
    echo "Skipping: Determine Camera Orientation"
  else
    section "Second Step: Determine Camera Orientation?"
    mm3d Tapas FraserBasic $imageRegex Out=$output
  fi

  #section "Generate Sparse Point Cloud"
  #mm3d AperiCloud $imageRegex $output

  #mm3d SaisieMasqQT "$mask.jpg" &&

  if [[ $skipTo > 3 ]]; then
    echo "Skipping: Generate Dense Point Cloud"
  else
    section " Third Step: Generate Dense Point Cloud?"
    mm3d C3DC $c3cdMode $imageRegex $output ZoomF=8
  fi

  if [[ $skipTo > 4 ]]; then
    echo "Skipping: Generate Mesh"
  else
    section "Fourth Step: Generate Mesh"
    mm3d TiPunch C3DC_${c3cdMode}.ply Out=$mesh Pattern=$imageRegex Mode=$c3cdMode
  fi

  if [[ $skipTo > 5 ]]; then
    echo "Skipping: Texturise Mesh"
  else
    section "Fifth Step: Texturise Mesh"
    mm3d Tequila $imageRegex Ori-$output $mesh
  fi

  echo "Final Mesh: [${final}]"
  echo "Final point cloud: [C3DC_${c3cdMode}.ply]"

  END_TIME=$(date +%s)
  echo "It took $(($END_TIME - $START_TIME)) seconds to finish..."
#}
#echo "source ~/Thesis/micmac.sh">> ~/.zshrc

# Clean: delete all but this file (how to get thid file name?)
# rm  -r $(find . -name "*" ! -name "micmac.sh")

#mm3d Tapioca All ".*.jpg" 1500
#mm3d Tapas FraserBasic ".*.jpg" Out=hand
#mm3d AperiCloud ".*.jpg" hand
#mm3d C3DC BigMac ".*.jpg" hand ZoomF=8
#mm3d TiPunch C3DC_BigMac.ply Out=MeshHand Pattern=".*.jpg" Mode="BigMac"
#mm3d Tequila ".*.jpg" Ori-hand MeshHand
#
#find . -depth -name "*.JPG" -exec sh -c 'f="{}"; mv -- "$f" "${f%.JPG}.jpg"' \;
