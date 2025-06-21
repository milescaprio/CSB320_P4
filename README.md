# CSB_P3
Project 4 for CSB 320: Reinforcement Learing

Setup: 

`conda env create`

`conda activate csb320-p4`

run:
- bandit_monte_carlo.ipynb
- gridworld1.ipynb
  

<u>Hyperparameters that had the biggest effect on performance</u>

In this test, I felt like the epsilon related settings carried the most weight. Learners needed a balanced exploration rate and a well tuned decay otherwise they either never settled on a good path or stopped exploring too soon. Alpha and Gamma had secondary impact, but without the right epsilon strategy, the agent failed to learn effectively.

<u> How does epsilon decay influence exploration versus exploitation ?</u>

Epsilon decay governs how quickly the agent shifts from trying new actions to repeating high reward moves. A slower decay keeps the agent curious and creative longer which helps discover optimal routes, whereas a faster decay forces exploitation early on which can lock the agent into suboptimal behaviors if it never explored enough.

<u>What challenges did you face in tuning parameters</u>

In my opinion, the hardest part was combining different parameters. While I had a decent understanding of what each parameter contributed to, it was more difficult to understand what behaviors a model would exhibit when I changed multiple at a time, and this was hard to test. I made my best guesses, but they weren't perfect.

<u>Compare and contrast monte carlo and gridworld hyperparameter effects</u>

The multi-armed bandit problem was much simpler than gridworld because it didn’t involve planning across states. Each choice was independent, and rewards came right away, so tuning mostly focused on epsilon and how much the agent explored. In gridworld, though, actions had long-term consequences, so hyperparameters like gamma and alpha were critical. The agent had to learn how current moves affect future rewards, which made training more sensitive and required a better balance between all parameters, not just exploration.









