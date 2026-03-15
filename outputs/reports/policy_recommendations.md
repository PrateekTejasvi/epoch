# Campus Policy Recommendations (Data-Backed)

## Context
This report summarizes findings from a reproducible observational pipeline on student digital harm.
Interpretation is associative and predictive, not causal proof.

## Recommendation 1: Platform-Targeted Risk Outreach
Prioritize counseling and digital wellness outreach by platform-specific harm profile rather than usage-hours alone.
Evidence: WhatsApp shows the highest average Harm_Index (0.694) after preprocessing.

## Recommendation 2: Sleep-First Academic Support Protocol
Introduce sleep-protection interventions (quiet hours campaigns, sleep hygiene nudges, advisor check-ins) as the first response for heavy social-media users.
Evidence: the estimated mediated (indirect) effect of usage through sleep on academic harm is 0.1051; this indicates sleep is a measurable pathway, not only a side symptom.

## Recommendation 3: Persona-Segmented Intervention Tracks
Deploy separate intervention tracks for high-conflict users, hidden-addiction users, and high-usage but resilient users instead of one broad program.
Evidence: The most common profile is The Social Warrior (236 students; 33.5%).

## Validation Snapshot
- Harm-index RMSE (Linear): 0.2382
- Harm-index RMSE (Tree Model): 0.1923
- Best predictive model for this run: gradient_boosting_fallback
