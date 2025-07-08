## 🏏 About This Project: Powerplay-Based IPL Match Winner Prediction

Hey Cricket Enthusiasts! 👋
Here's one of my favorite projects — predicting the **IPL Match winner right after the second innings Powerplay**, using just the Powerplay stats from both teams!

At first, the idea felt like a long shot. “**Can we really predict the match outcome just after the first 6 overs of the second innings?**” or **Am I Overthinking?** because it's Natural for me 🤷‍♂️. Even I had doubts. But after digging into multiple IPL seasons and closely observing match trends, I discovered that Powerplay performance often sets the tone for the rest of the game. 

---

### 🔍 How It All Started

I began with basic features like Powerplay scores for both Innings, wickets, toss outcome, and venue. While this gave me a good starting point, I realized the data needed more context.

That’s when I added the **target score for the chasing team** — unlocking key derived features like:

1. ***Powerplay run rate of the Team Batting first***
2. ***Required run rate after Powerplay for the Team Batting Second***
3. ***Wicket fall rate for Both Innings during Powerplays***
4. ***Venue-wise average Powerplay scores over the Seasons***

These helped reveal critical match dynamics during those crucial early overs.

---

### ⚙️ Model & Results

After experimenting with various ML models, I found **XGBoost Classifier** to perform the best, achieving around **79% accuracy** — just after the second innings Powerplay!

---

## 🎯 Why This Project Stands Out

Unlike most models that need full match or ball-by-ball data, this one provides an **Early prediction Snapshot** using only Powerplay data — making it suitable for **Real-Time Prediction**, and fan engagement.

---

💬 Feel free to explore the repo, raise issues, or connect if you're passionate about cricket + data. Contributions and feedback are always welcome!

---

For More Information, Please Head over to this Detailed Blog Published on Medium: https://ai.plainenglish.io/predicting-ipl-match-outcomes-using-powerplay-scores-and-machine-learning-62c1070da227
