# Explainable AI (XAI) Report: IntelliMoE Trace

This report documents the explainable decisions, routing logic, execution path, and API selections resolved for the latest query processed by IntelliMoE.

---

## 👤 User Request
> **i want know the yesterday odi match who was win**

---

## ⛓️ Execution Flow Diagram

```text
User Query: "i want know the yesterday odi match w..."
   ↓
ML Classifier Prediction: NEWS (Confidence: 28.8%)
   ↓
Decision Engine (Single Expert): NEWS
   ↓
API Inference: NEWS via Gemini
   ↓
Response Aggregator
   ↓
Answer Output

```

---

## 🧠 Reasoning Timeline Traces

### 💡 Step 1
ML Classifier predicted 'NEWS' with 28.8% confidence.

### 💡 Step 2
Confidence score threshold set to 60.0%.

### 💡 Step 3
No fallback required. ML Prediction selected as primary.

### 💡 Step 4
AI Decision Engine verified Primary Expert 'NEWS' alone is sufficient.
Reasoning: Single expert routing sufficient. Decision Engine bypassed for latency optimization.

### 💡 Step 5
API Selection resolved to: Gemini (gemini-3.1-flash-lite).
Justification: 'NEWS' query leverages Gemini's deep reasoning capabilities.



---

## ⏱️ Performance & Telemetry Summary

- **Total Execution Response Time**: 21910.43 ms
- **Estimated Token Volume**: 13 tokens
- **Fallback Triggered**: False
- **API Provider Endpoint(s)**: Gemini (gemini-3.1-flash-lite)

---

## 📰 News Live Search & Rewriter Details

- **Original User Query**: `i want know the yesterday odi match who was win`
- **Rewritten Search Query**: `Who won yesterday's ODI cricket match?`
- **Search Provider Used**: `Tavily Live Search`
- **Search Latency**: 3.31s
- **LLM Synthesis Latency**: 2.40s
- **Sources Used**: apnews.com, www.seattlepi.com, telanganatoday.com, www.reuters.com, www.bbc.com, abcnews.com, www.greenwichtime.com, www.beaumontenterprise.com, www.theguardian.com

### Retrieved Articles:

1. **[England vs India: Ben Duckett century and record Lord's ODI total help seal series win for hosts - BBC](https://www.bbc.com/sport/cricket/articles/cglj4n91744o)** (www.bbc.com)
   > **England hit the highest one-day international total at Lord's to beat India by 27 runs and seal a thrilling 2-1 series victory.**. The pair started swiftly, their 27 off the first three overs the highest score a team has ever reached in an ODI at Lord's, setting a platform for England's first century opening stand since the 2023 World Cup. They pushed on deep into the innings, with England not losing a wicket in the first 20 overs for the first time since their victorious 2019 World Cup. The records continued to tumble, with their partnership becoming England's highest opening stand at Lord's and their highest ever against India. Kohli, who at 37 may also be playing his final game at Lord's, made his first international half-century at the ground, but he was ultimately one of four batters to be dismissed by Curran, who profited as India desperately chased the game.

2. **[England win toss, bat in series-deciding 3rd ODI against India at Lord’s - AP News](https://apnews.com/article/england-india-cricket-odi-series-lords-root-7db97aec761751c070a8bd10f4c167a3)** (apnews.com)
   > Test Your News I.Q. 2026 Elections Election Results Election calendar White House Congress Supreme Court The latest AP-NORC polls. Your home base for in-depth reporting from the world of sports. Founded in 1846, AP today remains the most trusted source of fast, accurate, unbiased news in all formats and the essential provider of the technology and services vital to the news business. India players stand for a minute of silence in tribute to late cricketer Garfield Sobers at the start of the third ODI between England and India, at Lord’s cricket ground, London, Sunday, July 19, 2026. England players stand for a minute of silence in tribute to late cricketer Garfield Sobers at the start of the third ODI between England and India, at Lord’s cricket ground, London, Sunday, July 19, 2026. LONDON (AP) — England won the toss and opted to bat in the series-deciding third one-day international against India at Lord’s on Sunday.

3. **[Duckett's big century sets up England series win over India - Reuters](https://www.reuters.com/sports/cricket/ducketts-big-century-sets-up-england-series-win-over-india-2026-07-19/)** (www.reuters.com)
   > ## Browse World. Item 1 of 5 Cricket - Third One Day International - England v India - Lord's Cricket Ground, London, Britain - July 19, 2026 England's Ben Duckett celebrates with Joe Root after reaching his century Action Images via Reuters/Matthew Childs. **[1/5]**Cricket - Third One Day International - England v India - Lord's Cricket Ground, London, Britain - July 19, 2026 England's Ben Duckett celebrates with Joe Root after reaching his century Action Images via Reuters/Matthew Childs Purchase Licensing Rights, opens new tab. LONDON, July 19 (Reuters) - England beat India by 27 runs in a high-scoring contest at Lord's on Sunday to secure their first bilateral one-day international series victory against the world’s number ​one-ranked side since 2018. It was the highest innings in an ODI at Lord’s, eclipsing Viv Richards’s 138 not out in the 1979 World Cup final, and helped England to a colossal ​total of 387-3 in their 50 overs. *   About Reuters, opens new tab.

4. **[England wins ODI series against India after 747 runs scored in decider at Lord's - Greenwich Time](https://www.greenwichtime.com/sports/article/england-win-toss-bat-in-series-deciding-3rd-odi-22351178.php)** (www.greenwichtime.com)
   > # England wins ODI series against India after 747 runs scored in decider at Lord's. LONDON (AP) — Ben Duckett and Jacob Bethell shared a 192-run opening stand as England scored 387-3 and beat India by 27 runs in a high-scoring third ODI at Lord's on Sunday to win the cricket series 2-1. England and India piled up a jumbo-sized 747 runs between them as the visitors' defiant chase finished on 360-7, with Rohit Sharma scoring 138. Joe Root again impressed with a 48-ball 74 not out and Jos Buttler needed only 13 balls for his undefeated 41, with four fours and three sixes, as England amassed the highest ODI total at Lord’s. Sharma became the first India player to score an ODI hundred at Lord's in men's cricket — his 34th ton overall in the format — but he was bowled after England captain Harry Brook turned to Bethell to flip the momentum at 260-2 in the 39th over.

5. **[New Zealand beats West Indies in a 1-wicket thriller to clinch ODI cricket series - Greenwich Time](https://www.greenwichtime.com/sports/article/new-zealand-beats-west-indies-in-a-1-wicket-22351623.php)** (www.greenwichtime.com)
   > # New Zealand beats West Indies in a 1-wicket thriller to clinch ODI cricket series. BRIDGETOWN, Barbados (AP) — Mark Chapman scored 80 and Mitchell Santner shepherded the tail to guide New Zealand to a tense one-wicket win over West Indies in the fourth one-day cricket international Sunday and to an unassailable 3-1 lead in the five-match series. Santner was dropped with three runs needed and one wicket in hand, and last batter Jayden Lennox survived reviews for caught behind and lbw, before New Zealand edged past West Indies' modest total of 188 with 35 balls remaining. Santner, the New Zealand captain, hit the winning runs from the first ball of the 45th over to finish 34 not out and to spoil West Indies' hopes of achieving a series-leveling victory to pay tribute to cricket great Garry Sobers. West Indies had expressed a fervent hope to honor Sobers in his home town of Bridgetown and at Kensington Oval, the stadium where he nurtured his career and made his name.

6. **[England win toss, bat in series-deciding 3rd ODI against India at Lord's - Beaumont Enterprise](https://www.beaumontenterprise.com/sports/article/england-win-toss-bat-in-series-deciding-3rd-odi-22351178.php)** (www.beaumontenterprise.com)
   > Bridge City man dies after truck crash pushes van into house. # England win toss, bat in series-deciding 3rd ODI against India at Lord's. LONDON (AP) — England won the toss and opted to bat in the series-deciding third one-day international against India at Lord's on Sunday. England lost the opener at Edgbaston but bounced back in Cardiff, where Joe Root's 99 not out was the difference. England have not beaten India in an ODI series since 2018. The home side brought in Josh Tongue for Saqib Mahmood. Beaumont inspection scores range from A to C grades. Beachcombing Report: Messages from the Sea. One of the most exciting discoveries during my weekly beachcombing surveys is finding a message in... The Beaumont city council will consider several items at its next council meeting on Tuesday,... A Bridge City man died after a pickup truck crash on TX 62 sent his van into a house, leaving a... Beaumont man indicted after alleged assault, child endangerment.

7. **[England win toss, bat in series-deciding 3rd ODI against India at Lord's - Seattlepi.com](https://www.seattlepi.com/sports/england-win-toss-bat-in-series-deciding-3rd-odi-a22351178)** (www.seattlepi.com)
   > # England win toss, bat in series-deciding 3rd ODI against India at Lord's. LONDON (AP) — England won the toss and opted to bat in the series-deciding third one-day international against India at Lord's on Sunday. England lost the opener at Edgbaston but bounced back in Cardiff, where Joe Root's 99 not out was the difference. England have not beaten India in an ODI series since 2018. The home side brought in Josh Tongue for Saqib Mahmood. India made three changes with KL Rahul, Prince Yadav and Arshdeep Singh replacing Jasprit Bumrah, Washington Sundar and Shivam Dube. England: Ben Duckett, Jacob Bethell, Joe Root, Harry Brook (captain), Jos Buttler, Sam Curran, Will Jacks, Gus Atkinson, Jofra Archer, Adil Rashid, Josh Tongue. India: Rohit Sharma, Shubman Gill (captain), Virat Kohli, Ishan Kishan, Shreyas Iyer, KL Rahul, Axar Patel, Gurnoor Brar, Prince Yadav, Arshdeep Singh, Prasidh Krishna. The World Cup final will be played on turf with mixed reviews.

8. **[England win toss, bat in series-deciding 3rd ODI against India at Lord's - ABC News - Breaking News, Latest News and Videos](https://abcnews.com/Sports/wireStory/england-win-toss-bat-series-deciding-3rd-odi-134888250)** (abcnews.com)
   > # England win toss, bat in series-deciding 3rd ODI against India at Lord's. England have won the toss and opted to bat in the series-deciding third one-day international against India at Lord’s. LONDON -- England won the toss and opted to bat in the series-deciding third one-day international against India at Lord's on Sunday. England lost the opener at Edgbaston but bounced back in Cardiff, where Joe Root's 99 not out was the difference. England have not beaten India in an ODI series since 2018. The home side brought in Josh Tongue for Saqib Mahmood. India made three changes with KL Rahul, Prince Yadav and Arshdeep Singh replacing Jasprit Bumrah, Washington Sundar and Shivam Dube. ## Iran live updates: Kuwait, Bahrain report Iranian attacks following latest US strikes. * Jul 17, 10:29 AM. ## US conducts new strikes on Iranian targets, officials say. ### ABC News Live. 24/7 coverage of breaking news and live events.

9. **[England take India ODI series after Duckett’s ton and Bethell’s brilliant star turn - The Guardian](https://www.theguardian.com/sport/2026/jul/19/england-take-india-odi-series-after-ducketts-ton-and-bethells-brilliant-star-turn)** (www.theguardian.com)
   > # England take India ODI series after Duckett’s ton and Bethell’s brilliant star turn. In the end, despite a ferocious century from Rohit Sharma that turned Lord’s into his personal playground, England had enough on the board to seal a 2-1 series victory against India. Harry Brook must have thought it was mission accomplished for his side at the halfway stage, with Ben Duckett’s 141 from 135 balls having powered England to 387 for three. That was until Jacob Bethell, fresh from 91 with the bat, finally slid one of his left-arm tweakers past a sweep shot from Sharma and into the stumps to leave India at 260 for two in the 39th over. Still, as one half of a 192-run ­opening stand with Duckett, Bethell had helped to set a record by an ­England pair against India – any wicket – and the platform for this monster total.

10. **[Axar Patel credits patience and self-belief after match-winning all-round show against England - Telangana Today](https://telanganatoday.com/axar-patel-credits-patience-and-self-belief-after-match-winning-all-round-show-against-england)** (telanganatoday.com)
   > Home | Cricket | Axar Patel Credits Patience And Self Belief After Match Winning All Round Show Against England. India all-rounder Axar Patel said patience, self-belief and trusting the conditions were crucial to his match-winning performance in the first ODI against England. Axar starred with four wickets and an unbeaten half-century as India secured a six-wicket win to take a 1-0 series lead. **New Delhi:** India’s match-winner Axar Patel said patience, self-belief and trusting the conditions were the key factors behind his all-round display in the opening ODI against England at Edgbaston, where his four-wicket haul and unbeaten half-century helped India register a six-wicket victory and take a 1-0 lead in the three-match series. I feel like I was trying to hit the ball too hard during the T20Is. When you go in to bat in the death overs, you don’t have any other option but to go for big shots, but I was losing my shape a little.

