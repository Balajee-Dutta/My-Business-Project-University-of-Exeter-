# HR Assistant - 38 Strategic Evaluation Pairs Dataset
# Generated: 2025-08-14
# Total Evaluation Pairs: 38
# Distribution: Perfect (11), Good (11), Acceptable (6), Poor (10)
# 
# Column Definitions:
# job_id: Job identifier (JOB-001 to JOB-030)
# resume_id: Resume identifier (RES-001 to RES-030)
# overlap: Number of overlapping skills between resume and job
# human_rank: 1=Perfect Match, 2=Good Match, 3=Acceptable, 4=Poor Match
# match_type: Description of match scenario for analysis
#
# Business Purpose:
# - Test model accuracy across different matching scenarios
# - Evaluate perfect skill-experience alignment detection
# - Measure overqualified/underqualified candidate handling
# - Validate cross-functional transfer potential assessment
# - Assess domain mismatch identification capability
# - Enable Top-K accuracy measurements (Top-1, Top-3, Top-5, Top-10)
#
# Evaluation Strategy:
# - Perfect Matches (Rank 1): Should score highest similarity
# - Good Matches (Rank 2): Slight experience/skill differences
# - Acceptable (Rank 3): Overqualified or transferable skills
# - Poor Matches (Rank 4): Underqualified or domain mismatches