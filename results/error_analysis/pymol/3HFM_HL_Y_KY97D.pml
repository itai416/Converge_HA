# 3HFM_HL_Y KY97D: observed ddG 6.77, C: ridge predicted 1.16 kcal/mol
# residue numbers are those of the SKEMPI-renumbered file; run from this folder: pymol <this file>
load ../../../data/SKEMPI2_PDBs/PDBs/3HFM.pdb, cplx
hide everything
bg_color white
select antibody, chain H+L
select antigen, chain Y
show cartoon, antibody or antigen
set cartoon_transparency, 0.6
color grey80, antibody
color lightblue, antigen
select site, chain Y and resi 97
select partner_near, byres ((chain H+L) within 4.5 of site)
show sticks, site or partner_near
util.cbao site
util.cbag partner_near
distance polar, site, partner_near, 3.5, mode=2
label site and name CA, '%s%s' % (resn, resi)
label partner_near and name CA, '%s%s.%s' % (resn, resi, chain)
zoom site or partner_near, 4
deselect
