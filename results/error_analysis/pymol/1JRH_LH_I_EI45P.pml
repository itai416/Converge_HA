# 1JRH_LH_I EI45P: observed ddG -3.79, C: ridge predicted 0.85 kcal/mol
# residue numbers are those of the SKEMPI-renumbered file; run from this folder: pymol <this file>
load ../../../data/SKEMPI2_PDBs/PDBs/1JRH.pdb, cplx
hide everything
bg_color white
select antibody, chain L+H
select antigen, chain I
show cartoon, antibody or antigen
set cartoon_transparency, 0.6
color grey80, antibody
color lightblue, antigen
select site, chain I and resi 45
select partner_near, byres ((chain L+H) within 4.5 of site)
show sticks, site or partner_near
util.cbao site
util.cbag partner_near
distance polar, site, partner_near, 3.5, mode=2
label site and name CA, '%s%s' % (resn, resi)
label partner_near and name CA, '%s%s.%s' % (resn, resi, chain)
zoom site or partner_near, 4
deselect
