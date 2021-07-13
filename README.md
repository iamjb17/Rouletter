# Rouletter
Roulette predictor-

  The Rouletter program’s purpose is to help Betters Inc. be more efficient and effective when playing the casino game, roulette. It accomplishes this by attempting to predict which part of the wheel the ball will land in when it comes to a stop. Doing this effectively reduces the casino’s edge to below 0%. This means that the player now has an edge over the casino and if that edge holds up, is guaranteed to have positive returns given enough bets.
With this expectation in mind, according to our initial analysis, we have successfully been able to predict the correct outcomes with an average success rate of 40%. This is +6.667% more than the break-even point of 33.33%.

This outcome was achieved with these technical feats:
- A computer vision system that can watch a video of a roulette table in play and ID the ball and the “0” separately.
- An AI tracking system that can calculate the speed of the ball and the “0” by counting milliseconds per revolution of each.
- Utilization of neural network algorithms that can ingest this data and correlate it with the endpoint of the ball.
- A GUI, that allows the user to easily manipulate the data and make individual predictions based on real-time data. All within 8 seconds before the dealer ends the betting process.
-The use of charts and graphs makes it easy for the user to understand the summation of the neural network output and model performance.

### Note: Files ommitted from github:
- YOLOv5 computer vision AI
- PyQt5 GUI structure files
- Datasets used to train, validate, create models.

