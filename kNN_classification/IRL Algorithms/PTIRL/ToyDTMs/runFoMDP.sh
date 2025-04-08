data_location="/home/keumjoo/src/C++/DTM/ToyDTMs3/FoMDP_rest/P3/"
data_location="FoMDP-test/"
action_loc=${data_location}actions.csv
attribute_loc=${data_location}attributes.csv
target_loc=${data_location}lista.files
nontarget_loc=${data_location}listb.files
output_loc=${data_location}output
echo "python3.7 ./runFoMDP.py ${action_loc} ${attribute_loc} ${target_loc} ${nontarget_loc} ${output_loc}"
python3.7 ./runFoMDP.py ${action_loc} ${attribute_loc} ${target_loc} ${nontarget_loc} ${output_loc}
