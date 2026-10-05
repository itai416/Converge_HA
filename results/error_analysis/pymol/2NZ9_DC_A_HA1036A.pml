# 2NZ9_DC_A HA1036A: observed ddG 7.42, C: ridge predicted 1.32 kcal/mol
# residue numbers are those of the SKEMPI-renumbered file; run from this folder: pymol <this file>
load ../../../data/SKEMPI2_PDBs/PDBs/2NZ9.pdb, cplx
hide everything
bg_color white
select antibody, chain D+C
select antigen, chain A
show cartoon, antibody or antigen
set cartoon_transparency, 0.6
color grey80, antibody
color lightblue, antigen
select site, chain A and resi 1036
select partner_near, byres ((chain D+C) within 4.5 of site)
show sticks, site or partner_near
util.cbao site
util.cbag partner_near
distance polar, site, partner_near, 3.5, mode=2
label site and name CA, '%s%s' % (resn, resi)
label partner_near and name CA, '%s%s.%s' % (resn, resi, chain)
zoom site or partner_near, 4
deselect
