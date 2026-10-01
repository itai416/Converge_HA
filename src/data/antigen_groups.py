"""Antigen group per SKEMPI AB/AG complex, used to keep related complexes together in CV splits.

Hand-written because the `Protein 1/2` names are inconsistent (antigen can be either column,
same protein spelled several ways). Judgement calls are commented.
"""

ANTIGEN_GROUP = {
    # hen egg-white lysozyme (HyHEL-10/5/63, D1.3, D44.1 antibodies)
    **dict.fromkeys(
        ["1DQJ_AB_C", "1KIP_AB_C", "1KIQ_AB_C", "1KIR_AB_C", "1MLC_AB_E", "1VFB_AB_C", "1XGP_AB_C",
         "1XGQ_AB_C", "1XGR_AB_C", "1XGT_AB_C", "1XGU_AB_C", "1YQV_HL_Y", "3HFM_HL_Y"], "lysozyme"),
    **dict.fromkeys(["1BJ1_HL_VW", "1CZ8_HL_VW", "3BDY_HL_V"], "VEGF"),
    **dict.fromkeys(["1MHP_HL_A", "2B2X_HL_A"], "integrin_alpha1"),
    **dict.fromkeys(["1N8Z_AB_C", "3BE1_HL_A", "3N85_A_LH"], "HER2"),
    **dict.fromkeys(["1NCA_N_LH", "1NMB_N_LH"], "neuraminidase_N9"),
    **dict.fromkeys(["1YY9_CD_A", "4KRL_A_B", "4KRO_A_B", "4KRP_A_B"], "EGFR"),
    # 4JPK antigen (eOD-GT6) is an engineered gp120 outer-domain immunogen, grouped with gp120
    **dict.fromkeys(["2NY7_HL_G", "3IDX_HL_G", "3NGB_HL_G", "3SE8_HL_G", "3SE9_HL_G", "4JPK_HL_A"], "gp120"),
    **dict.fromkeys(["2NYY_DC_A", "2NZ9_DC_A"], "BoNT_A1"),
    # different influenza subtypes merged on purpose (conservative: fewer, larger groups)
    **dict.fromkeys(["2VIR_AB_C", "2VIS_AB_C", "3LZF_AB_HL", "4GXU_ABCDEF_MN", "4NM8_ABCDEF_HL"], "hemagglutinin"),
    **dict.fromkeys(["3BN9_B_CD", "3NPS_A_BC"], "MT-SP1"),
    **dict.fromkeys(["3G6D_LH_A", "3L5X_A_HL", "4I77_HL_Z"], "IL-13"),
    # the "antigen" is itself an antibody (D1.3 vs anti-idiotype E5.2): own class
    "1DVF_AB_CD": "anti_idiotype",
    "1AHW_AB_C": "tissue_factor",
    "1JRH_LH_I": "IFNg_receptor",
    "2BDN_HL_A": "MCP-1",
    "2JEL_LH_P": "HPr",
    "3W2D_A_HL": "SEB",
    "4U6H_AB_E": "vaccinia_L1",
    "4ZS6_HL_A": "MERS_spike",
    "5C6T_HL_A": "HCMV_gB",
    "5DWU_HL_AB": "cytokine_receptor_beta_c",
}
