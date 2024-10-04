# average_ldos
The script is used to produce an average local density of states colour map using CP2K. Therefore, it requires the outputs from CP2K and is not transferrable to other codes such as VASP or CASTEP. It does this by taking snapshots of an AIMD trajectory and running PBE0-TC-LRC DFT calculations on the snapshot geometry.

This script requires:
- local density of states for each snapshot
- projected density of states for each snapshot
- The geometry of each snapshot
- The hartree potential of each snapshot
- name of the files required: this is currently under the section labelled:
  ######################
  ### Key parameters ###
  ######################
- May need to update the atom_dict, to assign the elements used in your calculation a colour
- Currently has hardcoded bounaries of materials i.e. silica_min=37.5 and silica_max=47.5, this is specific to my interface and will need to be changed.

  How it works:
  - Enter in the type of interface looking at 9hardcoded to my system)
  - Enter the threshold value/ band tolerance. 
  - get_files functional searches through the directories in the path to ensure that the directory path has all the required files.
  - These directories are then used to get averaged values via the get_properties functional.
  - A smear is applied to the ldos with the functional km_smeared_ldos
  - The band edges i.e. the local homo/lumo or local vbm/cbm is assigned via the functional km_band_edge - This iterates through the ldos (currently mislabeled as pdos - sorry!) to meet the criteria set in lines 530 and 532 based on Fermi level and threshold value.
  - End of script is where the plots are made, currently there are quite a few different iterations of plots, I use the final plot which is saved as: plt.savefig('ldos_only_{0}.png'.format(figure_name),dpi=400) in my work.
