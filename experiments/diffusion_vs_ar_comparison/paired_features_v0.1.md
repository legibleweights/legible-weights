# GPT-2 vs MDLM — paired feature inspection

Eval set: 256 sequences × 256 tokens from OpenWebText (offset 40000, disjoint from training and correlation slices).

**A = GPT-2 small layer 6**, **B = MDLM-OWT layer 6**. Same tokenizer for both (GPT-2 BPE), so token strings are directly comparable.

## Top correlated pairs

### #1 — A feat 5280  ↔  B feat 277
act_corr = **0.989**, decoder_cos = -0.016

| rank | GPT-2 act | GPT-2 context | MDLM act | MDLM context |
|------|-----------|---------------|----------|--------------|
| 1 | 37.21 |  of posthegemony as a whole.**[ Which]** is fair enough, but a bit of | 69.71 |  photo or video image is an essential element**[ which]** should be integral part of any analysis, |
| 2 | 34.53 |  to businesses or consumers on Main Street.**[ Which]** raises the question, what is it intended | 69.14 | N1 Swine Flu global pandemic**[ which]** according to WHO and The CDC has killed |
| 3 | 31.56 |  acts a certain way. Precisely**[ which]** behavioral patterns or appearances are sufficient to sign | 68.15 |  work has been completed in all these projects**[ which]** are stalled as contractors have run out |
| 4 | 31.03 |  a lot of mathematical secrets in it,**[ which]** we don��t know how to | 66.88 |  we both have to open a bank account**[ which]** is accompanied by a mountain of paperwork to |

### #2 — A feat 2870  ↔  B feat 1617
act_corr = **0.988**, decoder_cos = -0.106

| rank | GPT-2 act | GPT-2 context | MDLM act | MDLM context |
|------|-----------|---------------|----------|--------------|
| 1 | 35.69 |  in Great Falls, Montana, will bear**[ the]** brunt of the impact.↵↵� | 98.04 | .↵↵Pour the mixture into**[ the]** skillet and spread it out with a spoon |
| 2 | 35.45 | �s Daily that failure would lead to**[ the]** party��s downfall.↵↵ | 95.79 | —the thicker (and more robust)**[ the]** membrane, the slower the ions will flow |
| 3 | 35.13 |  Oklahoma Attorney General Scott Pruitt. And by**[ the]** fact Trump is considering former Texas Gov. | 94.98 |  on the planet - simply by turning on**[ the]** TV.↵↵At the same time |
| 4 | 34.95 | ade parents from hovering over their children for**[ the]** next four years — interfering with the m | 94.42 | And both versions look better than**[ the]** PC version.↵↵For the most |

### #3 — A feat 7147  ↔  B feat 6916
act_corr = **0.987**, decoder_cos = +0.011

| rank | GPT-2 act | GPT-2 context | MDLM act | MDLM context |
|------|-----------|---------------|----------|--------------|
| 1 | 31.07 | 's leagues had become a major worry.**[↵]**↵The exodus of players to Europe, | 78.52 | -stick spray and heat to medium.**[↵]**↵Pour the mixture into the skillet |
| 2 | 30.71 |  here in every riding across the country.**[↵]**↵"We are clearly targeting resources where | 75.94 |  kind are not an issue to you.**[↵]**↵However you're forced to be less |
| 3 | 29.85 | ," says ODOT spokesman Don Hamilton.**[↵]** | 74.82 |  one that he never expected to tell.**[↵]**↵"I was suffering mentally with bipolar |
| 4 | 29.85 |  Wiltshire, was then arrested.**[↵]**↵Ms | 74.50 |  a spoon to a pancake shape.**[↵]**↵Cook for a few minutes until you |

### #4 — A feat 4513  ↔  B feat 9389
act_corr = **0.986**, decoder_cos = -0.044

| rank | GPT-2 act | GPT-2 context | MDLM act | MDLM context |
|------|-----------|---------------|----------|--------------|
| 1 | 33.20 | Dear friends,↵↵**[With]** a laughing and a crying eye we have | 83.31 | . I find even more enjoyable the video**[ with]** the brilliant Lenin references. Nonetheless, this |
| 2 | 33.07 |  by James Hohmann)↵↵**[With]** Breanne Deppisch↵↵THE | 80.06 |  not have access to true anti tank options**[ with]** this officer to destroy it. Cookiezn |
| 3 | 33.05 |  change as well.��↵↵**[With]** his optimistic assessment, Obama sought | 76.15 |  by James Hohmann)↵↵**[With]** Breanne Deppisch↵↵THE |
| 4 | 32.52 |  lifted years of economic sanctions.↵↵**[With]** Italian businesses signing deals worth around 17 billion | 76.13 |  economists have been too pessimistic on Russia,**[ with]** drop in oil prices and increased competition from |

### #5 — A feat 4915  ↔  B feat 3866
act_corr = **0.985**, decoder_cos = -0.022

| rank | GPT-2 act | GPT-2 context | MDLM act | MDLM context |
|------|-----------|---------------|----------|--------------|
| 1 | 39.27 |  will find some way to spin this,**[ but]** what else can you call someone who goes | 79.32 |  yes, it's a dumb idea,**[ but]** just go with it. Now say you |
| 2 | 37.97 |  a whole. Which is fair enough,**[ but]** a bit of a leap.↵↵ | 76.73 |  Germany. There are a few more,**[ but]** not all other countries in the world agree |
| 3 | 37.86 |  I know that sounds like a pun,**[ but]** we literally have water in some slides now | 76.19 | . Normally the coastal areas are cooler,**[ but]** today was just the opposite. A weak |
| 4 | 36.17 |  including presidents, might approve assassination plots,**[ but]** they didn��t brag about | 76.13 | . Cannot but feel a little cheated,**[ but]** noone is forcing me to purchase anything |

### #6 — A feat 7022  ↔  B feat 12096
act_corr = **0.984**, decoder_cos = -0.002

| rank | GPT-2 act | GPT-2 context | MDLM act | MDLM context |
|------|-----------|---------------|----------|--------------|
| 1 | 41.43 | PIC: Fine Gael**[ come]** in for criticism for this ad about a | 71.42 |  to be the first out of more to**[ come]**, I��ll start it off |
| 2 | 39.12 |  way. Physicists are able to**[ come]** up with things that surprise the mathematicians | 71.17 | . Perhaps more important, sometimes big plays**[ come]** from the element of |
| 3 | 37.83 |  from your streaming revenues: So basically you**[ come]** to the house, you pay 600euro | 69.32 | . Perhaps more important, sometimes big plays**[ come]** from the element of surprise.↵↵ |
| 4 | 37.46 |  able to use the same basic platform to**[ come]** out with variation after variation of some pretty | 67.88 |  a white man. I know this might**[ come]** as a shock because people do not tell |

### #7 — A feat 10877  ↔  B feat 2829
act_corr = **0.984**, decoder_cos = -0.009

| rank | GPT-2 act | GPT-2 context | MDLM act | MDLM context |
|------|-----------|---------------|----------|--------------|
| 1 | 42.84 |  a number of weeks later.↵↵**[From]** a real world standpoint, this entry doubles | 85.39 |  idea that many people were dying each year**[ from]** the regular old flu. Shocking. |
| 2 | 41.84 |  Against His Net Neutrality Plan↵↵**[from]** the that's-a-temper | 82.25 |  now, like Honu:↵↵**[From]** Cabana Bay, we can see some |
| 3 | 41.80 |  in Fallout: New Vegas?↵↵**[From]** Oerjeke via Bethesda Blog↵ | 81.32 |  get more feedback on clinical sports medicine issues**[ from]** the readership. I plan over time |
| 4 | 41.10 |  now, like Honu:↵↵**[From]** Cabana Bay, we can see some | 81.00 |  the city of Humble, Texas,**[ from]** which he graduated in 1981,[7] |

### #8 — A feat 7489  ↔  B feat 1592
act_corr = **0.984**, decoder_cos = -0.024

| rank | GPT-2 act | GPT-2 context | MDLM act | MDLM context |
|------|-----------|---------------|----------|--------------|
| 1 | 33.79 |  it can perform memory operations.↵↵**[There]** is a set of processors that execute instructions | 105.21 |  Internal Revenue Service is happy to pretend that**[ there]** isn��t one, NPR� |
| 2 | 32.67 | Components of the System:↵↵**[There]** is a memory subsystem that supports the following | 104.32 | . Hold your head high, as if**[ there]** is a string attached |
| 3 | 32.41 |  Sheva reported.↵↵��**[There]** are 197 countries, like France, Algeria | 103.93 |  his contract at Borussia Dortmund means that**[ there]** will be more competition for places for Mand |
| 4 | 32.23 | , and water.↵↵��**[There]**��s no telling what you� | 103.19 |  little, insurance companies gave a little and**[ there]** would be enough to pay to get everyone |

### #9 — A feat 1270  ↔  B feat 7576
act_corr = **0.984**, decoder_cos = +0.039

| rank | GPT-2 act | GPT-2 context | MDLM act | MDLM context |
|------|-----------|---------------|----------|--------------|
| 1 | 35.76 |  a fan of the Disciples Series,**[ or]** you enjoy well written turn based strategy, | 104.24 |  in service, supplemented by diesel locomotive**[ or]** railcar.↵↵About↵↵ |
| 2 | 35.55 |  years, but never built a champion,**[ or]** even a playoff team. Most great players | 101.41 | , even in pristine areas in Scandinavia**[ or]** North America. This happens because most of |
| 3 | 35.17 |  the air and claimed that it was double**[ or]** triple this amount.↵↵Here is | 100.82 |  all of its new motors will be electric**[ or]** hybrid after 2019. But the ban wouldn |
| 4 | 34.97 |  whatever Trump happened to be thinking about,**[ or]** whatever he��d just seen while | 100.25 |  to back up and and go over this**[ or]** that item, and you get done with |

### #10 — A feat 910  ↔  B feat 4656
act_corr = **0.984**, decoder_cos = -0.056

| rank | GPT-2 act | GPT-2 context | MDLM act | MDLM context |
|------|-----------|---------------|----------|--------------|
| 1 | 33.23 |  don��t want to be imprisoned**[ by]** the past,�� Obama said during | 81.14 | s doing as it is a general recognition**[ by]** and that they needed to change things up |
| 2 | 32.72 |  the air of the troposphere is heated**[ by]** the earth��s surface. The | 76.87 | , the country will implode. Whether**[ by]** revolution or people simply sabotaging everything they |
| 3 | 32.50 | �t be able to avoid being erased**[ by]** history and the historic task the party carries | 76.58 | : Oklahoma Attorney General Scott Pruitt. And**[ by]** the fact Trump is considering former Texas Gov |
| 4 | 32.18 | speed gearbox that has been robotized**[ by]** Automac engineering. The dry weight is | 74.28 | per-second and scatter off each other**[ by]** the infinity-gazillions to create |

### #11 — A feat 9435  ↔  B feat 10042
act_corr = **0.983**, decoder_cos = -0.014

| rank | GPT-2 act | GPT-2 context | MDLM act | MDLM context |
|------|-----------|---------------|----------|--------------|
| 1 | 35.28 |  where he also teaches economics.↵↵**[When]** the presidents of colleges and universities talk privately | 81.07 | The party held just two seats in Parliament**[ when]** the election was called, but is fielding |
| 2 | 34.54 | The party held just two seats in Parliament**[ when]** the election was called, but is fielding | 80.59 | ↵The report says that in 2021,**[ when]** the authors predict that automated vehicles will be |
| 3 | 34.40 |  fans watch English games on TV↵↵**[When]** satellite television started broadcasting the top leagues of | 79.91 |  of minutes. Even a decade later,**[ when]** this image was taken, the radiation probably |
| 4 | 34.21 |  the one who would be crowned the champion**[ when]** everything was all said and done.↵ | 79.54 |  the one who would be crowned the champion**[ when]** everything was all said and done.↵ |

### #12 — A feat 6666  ↔  B feat 7532
act_corr = **0.983**, decoder_cos = +0.064

| rank | GPT-2 act | GPT-2 context | MDLM act | MDLM context |
|------|-----------|---------------|----------|--------------|
| 1 | 33.69 |  and he��s right. But**[ we]**��d argue that raising the minimum | 96.41 |  a cosmic answer but to comment on where**[ we]** are now. Physics in quantum field theory |
| 2 | 33.43 |  cut and thrust of politics.↵↵**[We]** met for a wide-ranging interview soon | 90.33 |  action within days.↵↵��**[We]** don��t want to be imprisoned |
| 3 | 32.33 | .↵↵But that is not how**[ we]** work, is it? I give you | 89.29 | not-leadership dept↵↵**[We]** were mystified last week when FCC chair |
| 4 | 32.21 |  Year Than In All Of 2016↵↵**[We]**��ve still got more than a | 89.28 |  and he��s right. But**[ we]**��d argue that raising the minimum |

## Divergent pairs (lowest correlation in the matching)

### #13 — A feat 9556  ↔  B feat 10456
act_corr = **0.039**, decoder_cos = -0.010

| rank | GPT-2 act | GPT-2 context | MDLM act | MDLM context |
|------|-----------|---------------|----------|--------------|
| 1 | 27.03 |  graduated. This spring the 6-foot**[-]**7, 293-pounder settled in | 23.02 |  to align their policy with the military's**[.]**<\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> |
| 2 | 22.53 | . This spring the 6-foot-**[7]**, 293-pounder settled in at | 22.28 |  Watch the festival live stream all weekend here**[.]**<\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> |
| 3 | 11.37 |  This spring the 6-foot-7**[,]** 293-pounder settled in at right | 21.78 | ↵Your SvoëMesto**[ Team]**<\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> |
| 4 | 10.50 | ↵Kris Humphries had 16 points**[ and]** 15 rebounds, following his strong performance against | 21.40 |  Assassin's Creed 4: Black Flag here**[.]**<\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> |

### #14 — A feat 12077  ↔  B feat 11768
act_corr = **0.042**, decoder_cos = +0.011

| rank | GPT-2 act | GPT-2 context | MDLM act | MDLM context |
|------|-----------|---------------|----------|--------------|
| 1 | 18.56 |  this knowledge deficit as an occasion to look**[ more]** carefully at the temperature history of the great | 22.44 |  Guy.↵↵from the Comcastic**[ de]**pt↵↵New FCC boss Ajit |
| 2 | 18.01 |  in particular, but against someone who looks**[ or]** acts a certain way. Precisely | 20.21 |  playing really well. They have players that**[ are]** playing really well, just they're trying |
| 3 | 17.85 |  tablet, which means the Surface 3 looks**[ a]** lot like the Surface 2. It maintains | 17.65 |  to upgrade an existing rail line. Not**[ only]** that, but he says ��This |
| 4 | 16.65 | ↵↵Overall, the game looks a**[ lot]** like its forefather Galactic Civilizations II | 17.34 |  Bible.�� There��s**[ no]** evidence that he said anything of the sort |

### #15 — A feat 2692  ↔  B feat 1405
act_corr = **0.043**, decoder_cos = +0.050

| rank | GPT-2 act | GPT-2 context | MDLM act | MDLM context |
|------|-----------|---------------|----------|--------------|
| 1 | 15.33 |  David Sanger reported in Friday��**[s]** Times that in his first weeks on the | 23.42 |  last October when it accurately predicted: "**[The]** harder the conflict, the more glorious the |
| 2 | 12.63 |  high level recruits in last year��**[s]** Class of 2015. One such prospect was | 19.41 |  seat. Put your feet up. This**[ may]** take some time. Can I get you |
| 3 | 12.53 |  and David Sanger reported in Friday�**[�]**s Times that in his first weeks on | 18.43 |  from BitcoinDark.↵↵Apologies**[ for]** the inconveniences.<\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> |
| 4 | 12.48 |  we need now is for Friday��**[s]** Non-Farm Payroll number to come | 18.36 | ermott↵↵Investigation: Schweizer**['s]** Facts " |

### #16 — A feat 10847  ↔  B feat 1889
act_corr = **0.043**, decoder_cos = -0.003

| rank | GPT-2 act | GPT-2 context | MDLM act | MDLM context |
|------|-----------|---------------|----------|--------------|
| 1 | 29.52 |  ceiling that would have let the government keep**[ borrowing]** as much as it needed, and said | 21.19 |  California Highway Patrol, who called the demonstration**[ large]** but peaceful and estimated the crowd at above |
| 2 | 28.73 |  a deficit that usually must be financed by**[ borrowing]**.↵↵Turkey's current account deficits | 19.41 | The campaign worker, who spoke on condition**[ that]** his name not be used, said he |
| 3 | 28.39 |  have some funds available with them to get**[ loans]** for new works. It is also expected | 17.02 |  applied at the query stage rather than the**[ collection]** stage, is defensible. But does |
| 4 | 22.75 |  use the library for traditional reasons: to**[ borrow]** printed books, or sit, read, | 16.96 |  in an SQL database, including the design**[ I]** call Closure Table.↵↵In |
