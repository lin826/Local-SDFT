"""Build geek-jokes training rows from PHD Comics metadata.

Field convention (Alpaca JSONL consumed by ``configs/geek_jokes.yaml``):

- **instruction** — fixed task: tell a spoken setup/punchline geek joke about grad school.
- **input** — ``Give a joke about {topic}.`` where *topic* is a lower-case phrase
  derived from the comic title / caption (never the fake journal abstract).
- **output** — three newline-separated lines:

  1. Setup question ending with ``?`` (invites "Why?" or a wrong guess).
  2. Interactive beat: ``Why?`` or ``Is it because …?``
  3. Punchline: ``Because …`` or ``No — …`` grounded in the comic's core idea.

Journal abstracts from phdcomics.com are intentionally **not** used as output.
"""

from __future__ import annotations

import re
from typing import NamedTuple

INSTRUCTION = (
    "Tell a geek joke about PhD or grad school life as a spoken setup/punchline joke."
)

_Joke = NamedTuple("_Joke", [("setup", str), ("beat", str), ("punchline", str)])


def _normalize_title(title: str) -> str:
    return title.replace("`", "'").strip()


def _topic_phrase(title: str) -> str:
    """Lower-case topic phrase for the ``input`` field."""
    t = _normalize_title(title)
    t = re.sub(r"\s+v\.\s+", " versus ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+vs\.?\s+", " versus ", t, flags=re.IGNORECASE)
    t = re.sub(r"^the\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+explained$", "", t, flags=re.IGNORECASE)
    t = re.sub(r",\s*pt\.\s*\d+(?:\s+of\s+\d+)?$", "", t, flags=re.IGNORECASE)
    return t.lower()


def build_input(title: str) -> str:
    return f"Give a joke about {_topic_phrase(title)}."


def _fmt(joke: _Joke) -> str:
    return f"{joke.setup}\n{joke.beat}\n{joke.punchline}"


# Curated setup/punchline jokes keyed by normalized comic title.
# Punchlines infer the comic's core idea from its title (PHD Comics themes).
_JOKES: dict[str, _Joke] = {
    "Graph - Work output": _Joke(
        "What does a grad student's work-output graph look like over the semester?",
        "Is it because it's a smooth upward trend like a Nature figure?",
        "No — it's flat for months, then a vertical cliff the night before the deadline.",
    ),
    "Grad student etiquette": _Joke(
        "What's the number-one rule of grad student etiquette?",
        "Is it because you should always bring snacks to lab meeting?",
        "No — never ask another grad student when they're going to finish.",
    ),
    "Vacation v. Stress": _Joke(
        'Why is a PhD student\'s "vacation" more stressful than the rest of the semester?',
        "Why?",
        "Because that's when their advisor finally has time to email them about revisions.",
    ),
    "How Grad School is just like Kindergarten": _Joke(
        "How is grad school just like kindergarten?",
        "Is it because everyone gets a gold star for trying?",
        "No — you still need permission slips, scheduled naps, and an adult to tell you when you can leave.",
    ),
    "Seminar Bingo": _Joke(
        "How do grad students survive a boring seminar?",
        "Is it because they take meticulous notes on the science?",
        "No — they play seminar bingo and hope nobody says \"in conclusion\" before they get five in a row.",
    ),
    "The Economic Meltdown": _Joke(
        "What happened to the department budget during the economic meltdown?",
        "Is it because they cut coffee first?",
        "No — they kept the coffee and cut everything that might accidentally fund your experiment.",
    ),
    "Author List": _Joke(
        "Why is the author list longer than the abstract?",
        "Is it because the science required a huge collaboration?",
        "No — because everyone who ever walked past the lab gets a middle-authorship for moral support.",
    ),
    "Marriage v. The Ph.D.": _Joke(
        "In the fight between marriage and the Ph.D., who wins?",
        "Is it because true love conquers all?",
        "No — the Ph.D. wins on points, on time, and usually on weekends too.",
    ),
    "Addressing Reviewer Comments": _Joke(
        "How do you address reviewer comments on your paper?",
        "Is it because you carefully re-run every experiment they asked for?",
        "No — you write a polite rebuttal explaining why the reviewer misunderstood their own field.",
    ),
    "Average time spent writing one e-mail": _Joke(
        "How long does it take a grad student to write one email to their advisor?",
        "Is it because they type fast?",
        "No — three hours: one hour drafting, one hour deleting, one hour rephrasing \"just checking in.\"",
    ),
    "Analysis of Value": _Joke(
        "How does a grad student perform an analysis of value?",
        "Is it because they calculate ROI on lab equipment?",
        "No — they weigh whether free pizza at the seminar is worth sitting through the talk.",
    ),
    "Abstract Mad Libs": _Joke(
        "How do you write a conference abstract in ten minutes?",
        "Is it because you're finally inspired?",
        "No — you play abstract Mad Libs and slot in \"novel,\" \"robust,\" and \"paradigm shift.\"",
    ),
    "Graph - Motivation level": _Joke(
        "What does the grad student motivation graph look like?",
        "Is it because it peaks right after passing quals?",
        "No — it starts high at orientation and asymptotically approaches zero until the stipend hits.",
    ),
    "Science News Cycle": _Joke(
        "What is the science news cycle?",
        "Is it because peer review keeps quality high?",
        "No — breakthrough, hype, backlash, \"actually we already knew that,\" repeat.",
    ),
    "Your Life Ambition": _Joke(
        "What happens to your life ambition in grad school?",
        "Is it because you refine it through thoughtful mentorship?",
        "No — it shrinks until your biggest dream is a desk by a window.",
    ),
    "We're all doomed": _Joke(
        "Why do grad students feel like we're all doomed?",
        "Why?",
        "Because the experiment failed, the deadline moved up, and the printer is out of toner — again.",
    ),
    "The Actual Method": _Joke(
        "What's the difference between the method section and the actual method?",
        "Is it because you cleaned up the prose for clarity?",
        "No — the paper says \"standard protocol\"; the actual method is panic, duct tape, and prayer.",
    ),
    "Writing your Thesis Outline": _Joke(
        "How far do you get writing your thesis outline?",
        "Is it because chapter one writes itself?",
        "No — you finish the outline, rename it notFinal.doc, and open a fresh blank document.",
    ),
    "Facebook": _Joke(
        "Why does a grad student's Facebook activity spike at 2 p.m. on a Tuesday?",
        "Is it because they're networking with colleagues?",
        "No — because that's when lab meeting ends and the guilt hasn't kicked in yet.",
    ),
    "Deciphering Academese": _Joke(
        "How do you decipher academese in a paper?",
        "Is it because you look up every jargon term?",
        "No — you translate \"further research is needed\" to \"we have no idea what this means either.\"",
    ),
    "Why are we doing this?": _Joke(
        "Why are we doing this PhD thing anyway?",
        "Why?",
        "Because at some point someone told us we were smart, and we believed them before seeing the job market.",
    ),
    "Vicious Cycle": _Joke(
        "What's the vicious cycle of grad school research?",
        "Is it because you iterate until the hypothesis is confirmed?",
        "No — experiment fails, tweak one variable, rerun, fail again, question life choices, repeat.",
    ),
    "The F-1 Process Explained": _Joke(
        "Can you explain the F-1 visa process for international students?",
        "Is it because it's straightforward once you read the forms?",
        "No — it's a labyrinth where every answer spawns three new forms and one anxious email.",
    ),
    "The Repulsor Field Explained": _Joke(
        "What is the advisor repulsor field?",
        "Is it because they're busy with important science?",
        "No — the closer you get with a question, the stronger the force pushing you toward \"email me later.\"",
    ),
    "Newton's Three Laws of Graduation": _Joke(
        "What are Newton's three laws of graduation?",
        "Is it because an object at rest stays at rest until comps?",
        "No — a thesis in motion stays in motion unless acted upon by a committee meeting.",
    ),
    "If TV Science was more like REAL Science": _Joke(
        "What if TV science were more like real science?",
        "Is it because every montage would show someone waiting for a centrifuge?",
        "No — the episode ends with \"results inconclusive, repeat for six months.\"",
    ),
    "Academic Salaries": _Joke(
        "Why do academic salaries look the way they do?",
        "Is it because universities pay for passion, not money?",
        "No — because they know you'll stay for the title and the free departmental mug.",
    ),
    "Intellectual Freedom": _Joke(
        "What does intellectual freedom mean in academia?",
        "Is it because you can pursue any research question?",
        "No — you're free to pursue any question that fits the grant you accidentally got.",
    ),
    "Red ink": _Joke(
        "Why is there so much red ink on your draft?",
        "Is it because your advisor is a meticulous editor?",
        "No — because \"looks good\" in academia still means \"rewrite everything in track changes.\"",
    ),
    "How Professors spend their time": _Joke(
        "How do professors actually spend their time?",
        "Is it because they're mentoring students and doing groundbreaking research?",
        "No — meetings, email, meetings about email, and a slide deck labeled \"quick update.\"",
    ),
    "Dating Odds": _Joke(
        "What are a grad student's dating odds?",
        "Is it because smart is the new sexy?",
        "No — you're competing with your thesis for attention, and the thesis is more demanding.",
    ),
    "notFinal.doc": _Joke(
        "Why is your thesis still called notFinal.doc?",
        "Why?",
        "Because every version is final until your advisor opens it and says it's not.",
    ),
    "Our Thesis": _Joke(
        "Whose thesis is it, really?",
        "Is it because it's a collaborative intellectual journey?",
        "No — it's \"our thesis\" until something breaks; then it's suddenly just yours.",
    ),
    "Undergradese": _Joke(
        "What is undergradese?",
        "Is it because freshmen speak a charming dialect?",
        "No — it's \"Will this be on the exam?\" translated into every office-hour question.",
    ),
    "World Cup vs. PhD": _Joke(
        "World Cup or PhD — which takes longer and hurts more?",
        "Is it because the World Cup only happens every four years?",
        "No — the PhD also takes four years, except yours adds extra time and no trophy.",
    ),
    "Post-Bachelors Disorder": _Joke(
        "What are the symptoms of post-bachelor's disorder?",
        "Is it because you miss having a normal sleep schedule?",
        "No — you can't stop saying \"when I was an undergrad\" and you flinch when someone says \"real job.\"",
    ),
    "The Grant Cycle": _Joke(
        "What is the grant cycle?",
        "Is it because funding follows merit?",
        "No — write proposal, get rejected, revise, get rejected softer, celebrate a pilot grant, repeat.",
    ),
    "Unemployment vs. Graduate Stipends": _Joke(
        "Unemployment or a graduate stipend — which pays better?",
        "Is it because stipends come with health insurance?",
        "No — unemployment sometimes pays more, and nobody asks you to TA at 8 a.m.",
    ),
    "Your Thesis Title": _Joke(
        "How do you pick your thesis title?",
        "Is it because it should capture the soul of your research?",
        "No — you cram as many buzzwords as possible before the colon and hope nobody reads past it.",
    ),
    "What to call your Academic Event": _Joke(
        "What do you call your academic event?",
        "Is it because \"symposium\" sounds prestigious?",
        "No — you add \"international\" and \"workshop\" until it fits on a flyer nobody attends.",
    ),
    "Valentine's day 2006": _Joke(
        "What does a grad student get for Valentine's Day?",
        "Is it because romance survives long lab hours?",
        "No — a card that says \"I love you almost as much as my negative controls worked.\"",
    ),
    "The Thesis Committee": _Joke(
        "What is a thesis committee really for?",
        "Is it because they guide your intellectual development?",
        "No — to ask if you've considered a completely different methodology three weeks before defense.",
    ),
    "Your Shrinking Sense of Humor": _Joke(
        "Why does your sense of humor shrink in grad school?",
        "Is it because the work is so serious?",
        "No — because you've laughed at the same lab joke for five years and now it's just muscle memory.",
    ),
    "Ninjas vs. Professors": _Joke(
        "Ninjas versus professors — who is stealthier?",
        "Is it because ninjas train for years in silence?",
        "No — professors disappear from email for weeks and reappear with \"quick thoughts.\"",
    ),
    "Your Computer Desktop": _Joke(
        "What does a grad student's computer desktop look like?",
        "Is it because it's organized by project?",
        "No — 400 files named final, final2, and FINAL_really_this_time scattered like landmines.",
    ),
    "Happening outside": _Joke(
        "What's happening outside the lab window?",
        "Is it because you haven't looked up from your data in days?",
        "No — sunshine, social life, and seasons changing while you calibrate the same instrument.",
    ),
    "Anatomy of a group meeting presentation": _Joke(
        "What is the anatomy of a group meeting presentation?",
        "Is it because you lead with results and close with impact?",
        "No — apologies for no data, excuses for broken equipment, and \"questions?\" spoken as a plea.",
    ),
    "Why? Why??": _Joke(
        "Why? Why are we still in grad school?",
        "Why?",
        "Because every time you try to leave, someone says you're so close — and you were close three years ago too.",
    ),
    "When to tell your advisor you're going on vacation": _Joke(
        "When should you tell your advisor you're going on vacation?",
        "Is it because transparency builds trust?",
        "No — after you've already left, from a timezone where replies take three business days.",
    ),
    "Macs": _Joke(
        "Why do Macs dominate academic laptops?",
        "Is it because they're better for science?",
        "No — because nothing says \"I'm writing my thesis at a café\" like a glowing apple and low battery.",
    ),
    "Things to do...": _Joke(
        "What's on a grad student's to-do list?",
        "Is it because it's prioritized and manageable?",
        "No — a list so long that \"things to do\" became its own permanent research project.",
    ),
    "Crushing the Dream": _Joke(
        "How does grad school crush the dream?",
        "Is it because reality tempers idealism?",
        "No — one comment at journal club and your Nobel speech becomes a nap fantasy.",
    ),
    "Fume Hood": _Joke(
        "Why is the fume hood the best place to hide?",
        "Is it because the ventilation muffles sound?",
        "No — because nobody looks for you where the chemicals are labeled \"unknown toxicity.\"",
    ),
    "Brain on a stick": _Joke(
        "What does a grad student look like after comps?",
        "Is it because you're sharpened and focused?",
        "No — a brain on a stick, caffeinated, and held together by sheer spite.",
    ),
    "How long your Prof. thinks it should take to do something": _Joke(
        "How long does your professor think a task should take?",
        "Is it because they remember doing it faster in their day?",
        "No — \"a quick experiment\" means two weeks, unless you're the one doing it; then one afternoon.",
    ),
    "Where do you sit?": _Joke(
        "Where do you sit in seminar?",
        "Is it because front row shows engagement?",
        "No — you map the seats by escape routes, outlet access, and distance from being called on.",
    ),
    "Desk Entropy": _Joke(
        "What is desk entropy in a grad student office?",
        "Is it because chaos breeds creativity?",
        "No — papers, coffee cups, and USB cables tend toward maximum disorder unless a visitor is coming.",
    ),
    "How your Conference Presentation Goes": _Joke(
        "How does your conference presentation go?",
        "Is it because you rehearsed and the data is solid?",
        "No — laptop adapter missing, slides from 3 a.m., and a question you answered in slide two.",
    ),
    "Grading Rubric": _Joke(
        "What's on the TA grading rubric?",
        "Is it because points are assigned fairly?",
        "No — half credit for trying, full credit if they spelled your name right on the eval.",
    ),
    "Peak Productivity": _Joke(
        "When is a grad student's peak productivity?",
        "Is it because morning people win in academia?",
        "No — thirty minutes before something is due, powered by fear and stale coffee.",
    ),
    "The Origin": _Joke(
        "Where did your research question originate?",
        "Is it because of a elegant theoretical framework?",
        "No — a weird result nobody wanted to throw away became a whole dissertation chapter.",
    ),
    "Needs work": _Joke(
        "What does \"needs work\" mean on your draft?",
        "Is it because a few paragraphs need polish?",
        "No — rewrite the introduction, redo the figures, and maybe change fields while you're at it.",
    ),
    "Take it out": _Joke(
        "What happens when your advisor says \"take it out\"?",
        "Is it because you delete one redundant sentence?",
        "No — you remove the best figure, the cleverest paragraph, and your favorite pun.",
    ),
    "Draft approved!": _Joke(
        "Your advisor says \"draft approved\" — now what?",
        "Is it because you submit immediately?",
        "No — you wait for the email that begins \"Actually, one small thing…\"",
    ),
    "63% of internet readers will like this comic": _Joke(
        "Why will 63% of internet readers like this comic?",
        "Is it because the sample size is robust?",
        "No — because 63% of grad students are procrastinating and will click anything with a graph.",
    ),
    "Relationship status": _Joke(
        "What's a grad student's relationship status?",
        "Is it because it's complicated?",
        "No — \"in a committed relationship with my laptop, it's open but mostly one-sided.\"",
    ),
    "Sentences": _Joke(
        "How do you write academic sentences?",
        "Is it because clarity is the goal?",
        "No — you nest so many clauses that the verb graduates before the subject finishes.",
    ),
    "Call for Papers!": _Joke(
        "Why does every listserv send a call for papers?",
        "Is it because the community is thriving?",
        "No — because deadline tomorrow, fee non-refundable, and your abstract is still Mad Libs.",
    ),
    "Guide to T.A. Office Hours": _Joke(
        "What is the TA office-hours survival guide?",
        "Is it because you prepare detailed answers?",
        "No — smile, nod, and say \"great question — let's look that up together\" until time runs out.",
    ),
    "Valentine 2007": _Joke(
        "What's the grad-school Valentine's gesture?",
        "Is it because you cook a nice dinner?",
        "No — you share the second monitor so you both can write grants on date night.",
    ),
    "Food Chain": _Joke(
        "What is the academic food chain?",
        "Is it because merit rises to the top?",
        "No — PI eats postdoc, postdoc eats grad student, grad student eats free pizza crusts.",
    ),
    "How to look busy": _Joke(
        "How do you look busy when the experiment is incubating?",
        "Is it because you read papers diligently?",
        "No — furious typing, intense frown, and a terminal full of commands you ran yesterday.",
    ),
    "How do I love you?": _Joke(
        "How do I love thee in grad school?",
        "Is it because passion conquers deadlines?",
        "No — let me count the ways: zero, unless you count shared stress-crying over revision requests.",
    ),
    "A list, plan or outline of things to be done": _Joke(
        "Why do you keep making lists of things to be done?",
        "Is it because planning equals progress?",
        "No — outlining the outline feels productive without touching the actual thesis.",
    ),
    "What you know vs How much you know about it": _Joke(
        "What you know versus how much you know about it — which wins in seminar?",
        "Is it because depth beats breadth?",
        "No — you know one narrow thing deeply and hope nobody asks about the rest of the field.",
    ),
    "Snooze": _Joke(
        "Why is the snooze button a grad student's best friend?",
        "Why?",
        "Because every nine minutes is another tiny thesis extension you grant yourself.",
    ),
    "Humanities vs. Social Sciences": _Joke(
        "Humanities versus social sciences — who has it harder?",
        "Is it because both suffer equally?",
        "No — they argue about theory while sharing the same broken departmental printer.",
    ),
    "Grading Homeworks": _Joke(
        "How long does grading homework take?",
        "Is it because rubrics make it fast?",
        "No — longer if you try to decipher handwriting that evolved its own alphabet.",
    ),
    "Demonstration": _Joke(
        "What happens during the live demonstration?",
        "Is it because you rehearsed the demo flawlessly?",
        "No — Murphy's law gets tenure and your control doesn't control anything.",
    ),
    "How To Write An E-mail To Your Instructor Or Teaching Assistant": _Joke(
        "How should undergrads write email to their TA?",
        "Is it because a polite greeting is enough?",
        "No — they skip your name, the course number, and any indication they've attended class.",
    ),
    "A story in file names": _Joke(
        "Can you tell a story from grad-school file names alone?",
        "Is it because the narrative arc is clear?",
        "No — draft, draft2, draft_final, draft_final_USE_THIS, and trash_me_maybe.docx.",
    ),
    "Regular Working Hours": _Joke(
        "What are regular working hours for a grad student?",
        "Is it because nine to five keeps you healthy?",
        "No — whenever the instrument is free, minus whenever you're pretending to have a life.",
    ),
    "Your Impact Factor": _Joke(
        "What is your personal impact factor?",
        "Is it because citations measure influence?",
        "No — number of times you made coffee before anyone noticed you were in the lab.",
    ),
    "ASAP!": _Joke(
        "Your PI emails \"ASAP!\" — what do you do?",
        "Is it because you drop everything immediately?",
        "No — you panic, finish what you were doing, then discover ASAP meant \"sometime this month.\"",
    ),
    "The Claus Hypothesis": _Joke(
        "What is the Claus hypothesis?",
        "Is it because it's a festive research framework?",
        "No — you only get results if you believe, and the review board still wants more data.",
    ),
    "Too due list": _Joke(
        "Why is your to-do list called the \"too due\" list?",
        "Why?",
        "Because everything on it was due yesterday and you're adding new items anyway.",
    ),
    "Negation Field": _Joke(
        "What is the negation field around your advisor?",
        "Is it because they're encouraging?",
        "No — every idea you bring in gets a forceful \"no\" until it becomes their idea.",
    ),
    "PhD Widows": _Joke(
        "Who are PhD widows?",
        "Is it because the degree takes a weekend?",
        "No — partners who haven't seen you at dinner since you said \"just one more experiment.\"",
    ),
    "Choice of words": _Joke(
        "Why does word choice matter so much in academia?",
        "Is it because precision prevents misunderstanding?",
        "No — \"interesting\" means terrible, \"preliminary\" means we have no idea, and \"novel\" means we hope.",
    ),
    "Your life": _Joke(
        "What happened to your life after starting grad school?",
        "Is it because you achieved work-life balance?",
        "No — your hobbies became conferences, your friends became labmates, and sleep became optional.",
    ),
    "Clarity and depth": _Joke(
        "Can you have both clarity and depth in a thesis chapter?",
        "Is it because good writing does both?",
        "No — pick one; the committee will ask for the other in revisions anyway.",
    ),
    "The Methodology Translator": _Joke(
        "What does the methodology translator do?",
        "Is it because it converts protocols across fields?",
        "No — it turns \"we tried stuff until something worked\" into a five-page methods section.",
    ),
    "I should be done in...": _Joke(
        "When will you be done with the PhD?",
        "Is it because you have a realistic timeline?",
        "No — \"I should be done in a year\" is a sentence you've said every year since year three.",
    ),
    "When to meet with your advisor": _Joke(
        "When is the best time to meet with your advisor?",
        "Is it because open-door policies help?",
        "No — when they're about to leave for a conference and can't assign new tasks.",
    ),
    "Life Plan vs. Life Reality": _Joke(
        "Life plan versus life reality — how do they compare for grad students?",
        "Is it because you adapt gracefully?",
        "No — the plan had a house and hobbies; reality has shared housing and a lukewarm microwave burrito.",
    ),
    "Exclusive focus": _Joke(
        "Why does grad school demand exclusive focus?",
        "Is it because deep work requires solitude?",
        "No — because your brain is 60% thesis, 30% impostor syndrome, and 10% remembering to eat.",
    ),
    "What you brought to seminar": _Joke(
        "What did you bring to seminar today?",
        "Is it because you prepared thoughtful questions?",
        "No — a laptop, low expectations, and the silent hope the speaker runs out of time.",
    ),
    "Grooming vs. Time in Grad School": _Joke(
        "Grooming versus time in grad school — what wins?",
        "Is it because self-care matters?",
        "No — the longer you're in, the more acceptable it is to wear the same hoodie to every defense.",
    ),
    "Interdiscipline": _Joke(
        "What is interdiscipline in academia?",
        "Is it because fields collaborate freely?",
        "No — you're interdisciplinary until hiring committees ask which box you fit in.",
    ),
    "Deciding what to wear": _Joke(
        "How do grad students decide what to wear?",
        "Is it because presentation matters?",
        "No — clean-ish T-shirt if someone's visiting the lab; otherwise, whatever survived the laundry pile.",
    ),
    "Finished Thesis": _Joke(
        "When is a thesis truly finished?",
        "Is it because you submit to the registrar?",
        "No — when you've stopped waking up thinking about figure 3.2 — so, never.",
    ),
    "Data: by the numbers": _Joke(
        "What does your data look like by the numbers?",
        "Is it because n is large and p is small?",
        "No — three good points, forty outliers, and one legend you still need to fix.",
    ),
    "Nature vs. Science, pt. 1": _Joke(
        "Nature versus Science — which journal wins?",
        "Is it because merit decides?",
        "No — whichever one your rival didn't get into first becomes your \"backup plan.\"",
    ),
    "The Daily Routine": _Joke(
        "What is a grad student's daily routine?",
        "Is it because it's structured and healthy?",
        "No — coffee, failed experiment, lunch guilt, more coffee, pretend to write, go home late.",
    ),
    "Grad School the Board Game": _Joke(
        "How do you win Grad School: The Board Game?",
        "Is it because you finish first?",
        "No — you lose a turn every funding cycle and the winner is whoever still has sanity tokens.",
    ),
    "References": _Joke(
        "Why do reference lists keep growing?",
        "Is it because you're thorough?",
        "No — every reviewer says \"cite my work\" and BibTeX silently judges your life choices.",
    ),
    "Tales from the Road - MD Anderson Cancer Center": _Joke(
        "What happens on the conference road trip?",
        "Is it because you network and learn?",
        "No — delayed flights, poster session panic, and free hotel apples for dinner.",
    ),
    "Prospective grad student": _Joke(
        "What do prospective grad students ask on the visit weekend?",
        "Is it because they probe the research deeply?",
        "No — \"How's the stipend?\" and \"Do people ever finish on time?\" with hopeful eyes.",
    ),
    "Accounts Payable": _Joke(
        "Why is accounts payable the final boss?",
        "Is it because reimbursements are simple?",
        "No — you need receipts from a lunch you ate in 2019 and three signatures from people on sabbatical.",
    ),
    "Great Tweets of Science": _Joke(
        "What qualifies as a great tweet of science?",
        "Is it because it communicates discovery?",
        "No — a meme about p-values that gets retweeted by someone with a verified flask icon.",
    ),
    "It's in the syllabus": _Joke(
        "The student says \"it's in the syllabus\" — what happened?",
        "Is it because they read it carefully?",
        "No — they didn't read it; they're guessing, and you wrote the syllabus at 1 a.m.",
    ),
    "I am a writing god!": _Joke(
        "When do you feel like a writing god?",
        "Is it because the words flow effortlessly?",
        "No — for ten minutes after one good paragraph, until you reread it tomorrow.",
    ),
    "Unused Icons": _Joke(
        "Why does your desktop have so many unused icons?",
        "Is it because you're organized?",
        "No — each one is software you installed for one figure and never opened again.",
    ),
    "Procrascorrelation": _Joke(
        "What is procrascorrelation?",
        "Is it because procrastination correlates with creativity?",
        "No — the closer the deadline, the stronger your urge to clean the lab fridge.",
    ),
    "Grad School Energy Levels": _Joke(
        "What happens to grad school energy levels over time?",
        "Is it because you pace yourself?",
        "No — year one is a sprint, year three is a nap, year five is caffeine as a food group.",
    ),
    "The Plans": _Joke(
        "What happened to all your carefully made plans?",
        "Is it because you adapted strategically?",
        "No — the experiment laughed at Plan A, and Plans B through F are just more coffee.",
    ),
    "May or may not apply to reality": _Joke(
        "When does the protocol apply to reality?",
        "Is it because science is reproducible?",
        "No — the footnote says results may or may not apply to reality, especially yours.",
    ),
    "Draft dodging": _Joke(
        "What is draft dodging in grad school?",
        "Is it because you avoid military service?",
        "No — you dodge sending your draft until every typo becomes a personality trait.",
    ),
    "What to call your Professor": _Joke(
        "What should you call your professor?",
        "Is it because first names build rapport?",
        "No — \"Professor\" until they say otherwise, then still \"Professor\" for three more years.",
    ),
    "Definition of Vacation": _Joke(
        "What is the grad-school definition of vacation?",
        "Is it because you stop working entirely?",
        "No — working somewhere with better Wi-Fi and worse guilt about not being in lab.",
    ),
    "Mind over matter": _Joke(
        "Can mind over matter fix a failed experiment?",
        "Is it because positive thinking helps?",
        "No — if you don't mind, the broken equipment doesn't matter — until grant renewal.",
    ),
    "How well do you know your Advisor?": _Joke(
        "How well do you know your advisor?",
        "Is it because you meet weekly?",
        "No — you know their coffee order better than their weekend plans, and they've seen you cry.",
    ),
    "Food pyramid": _Joke(
        "What is the grad student food pyramid?",
        "Is it because balanced nutrition matters?",
        "No — base layer coffee, middle layer ramen, apex is free seminar cookies.",
    ),
    "Paste together": _Joke(
        "How do you finish the figure when the data is messy?",
        "Is it because you re-run the experiment?",
        "No — you paste together panels until it looks intentional in Photoshop.",
    ),
    "E-mail Panic": _Joke(
        "Why does a new email from your advisor cause panic?",
        "Why?",
        "Because the subject line is just your name and a question mark.",
    ),
    "Allnighter": _Joke(
        "Why pull an all-nighter before the deadline?",
        "Is it because the muse visits at midnight?",
        "No — because you spent the week \"getting organized\" and the muse brought red ink.",
    ),
    "A friendly reminder": _Joke(
        "Your advisor sends a friendly reminder — what does it mean?",
        "Is it because they're kindly checking in?",
        "No — you were supposed to finish yesterday and they're being polite about the panic.",
    ),
    "What do you want to be?": _Joke(
        "What do you want to be when you graduate?",
        "Is it because the path is clear?",
        "No — employed, mostly, and slightly less confused than when you started.",
    ),
    "Your Math Skills": _Joke(
        "What happened to your math skills in grad school?",
        "Is it because you use them daily?",
        "No — you can derive anything except your own budget and how many years you've been here.",
    ),
    "Some helpful advice": _Joke(
        "What's the most helpful advice you got in grad school?",
        "Is it because mentors share wisdom freely?",
        "No — \"It'll get worse before it gets better\" — and they were only half joking.",
    ),
    "Fake Window": _Joke(
        "Why does the basement lab have a fake window?",
        "Is it because architects love whimsy?",
        "No — so you can pretend daylight exists while your circadian rhythm files for divorce.",
    ),
    "And after that?": _Joke(
        "You defend the thesis — and after that?",
        "Is it because life becomes simple?",
        "No — postdoc applications, moving boxes, and explaining your title to relatives at Thanksgiving.",
    ),
    "Gravitational Waves Explained": _Joke(
        "Can you explain gravitational waves like a grad student?",
        "Is it because you simplify the physics?",
        "No — ripples in spacetime, ripples in your funding, same anxiety, different scale.",
    ),
    "Sleep": _Joke(
        "When do grad students sleep?",
        "Is it because eight hours keeps you sharp?",
        "No — in fifteen-minute intervals between alarm snoozes and instrument timers.",
    ),
    "Lab Hazard Rating System": _Joke(
        "What is the lab hazard rating today?",
        "Is it because safety comes first?",
        "No — elevated risk of explosion if anyone asks how the experiment went.",
    ),
    "Ready, set...": _Joke(
        "You're at \"ready, set\" — so where's \"go\"?",
        "Is it because the starting gun fired?",
        "No — stuck on \"set\" until IRB, safety training, and the one part that's back-ordered.",
    ),
    "Existential Deconstruction": _Joke(
        "Why deconstruct your research existentially at 2 a.m.?",
        "Why?",
        "Because if the null hypothesis is true, maybe your career path is too.",
    ),
    "Cosmic Inflation Explained": _Joke(
        "How is cosmic inflation like grad school?",
        "Is it because everything expands smoothly?",
        "No — the universe inflated fast; so did your reading list and your impostor syndrome.",
    ),
    "The Neurobiology of Writing": _Joke(
        "What does neuroscience say about thesis writing?",
        "Is it because flow states are real?",
        "No — fight-or-flight activates whenever you open a blank document.",
    ),
    "The Joys of LDRs": _Joke(
        "What are the joys of long-distance relationships in grad school?",
        "Is it because absence makes the heart grow fonder?",
        "No — you bond over shared calendars and \"sorry, I'm in lab\" texts.",
    ),
    "Outside interests": _Joke(
        "Do grad students have outside interests?",
        "Is it because hobbies keep you sane?",
        "No — your outside interest is thinking about work while pretending to have hobbies.",
    ),
    "Scooped": _Joke(
        "What does it feel like to get scooped?",
        "Is it because science moves fast?",
        "No — someone publishes your idea while you're still fixing the typo in the methods.",
    ),
    "The Semiotics of Professor E-mail Signatures": _Joke(
        "What do professor email signatures really mean?",
        "Is it because titles convey status?",
        "No — twelve lines of awards so you know they're important before reading \"see me.\"",
    ),
    "Lab coat rationale": _Joke(
        "Why wear a lab coat if you're mostly at a computer?",
        "Is it because safety regulations require it?",
        "No — so you look like science is happening while you update your bibliography.",
    ),
    "Clever Acronyms": _Joke(
        "Why do grant proposals love clever acronyms?",
        "Is it because they aid memorability?",
        "No — if the acronym is cute enough, reviewers forget the budget doesn't add up.",
    ),
    "Sit up Straight": _Joke(
        "Why does your advisor tell you to sit up straight?",
        "Is it because posture helps focus?",
        "No — so you look alert while they're explaining why your project needs to pivot.",
    ),
    "Grades don't matter": _Joke(
        "They say grades don't matter in grad school — is that true?",
        "Is it because research is what counts?",
        "No — until you TA a course and discover undergrads still very much care.",
    ),
    "Campus architecture": _Joke(
        "Why is campus architecture so confusing?",
        "Is it because buildings grow organically?",
        "No — so visiting parents get lost and never find the lab you're supposedly always in.",
    ),
    "Holiday!": _Joke(
        "What does a holiday mean for a grad student?",
        "Is it because you finally rest?",
        "No — the library is closed, which somehow makes you feel guilty for not working.",
    ),
    "Deadline": _Joke(
        "What is a deadline in grad school?",
        "Is it because it's a firm cutoff?",
        "No — a suggestion that becomes real only when someone else needs your PDF.",
    ),
    "Postcard": _Joke(
        "Why send a postcard from the conference?",
        "Is it because you're sightseeing?",
        "No — to prove you left the lab, even though you mostly networked in the poster hall.",
    ),
    "The Lab Hierarchy": _Joke(
        "What is the lab hierarchy?",
        "Is it because merit determines rank?",
        "No — PI, postdoc, grad student, rotating student, and the machine everyone fights over.",
    ),
    "What do you do?": _Joke(
        "What do you tell people you do in grad school?",
        "Is it because \"research\" is clear?",
        "No — you say \"I'm in grad school\" and watch their eyes glaze over politely.",
    ),
    "An Honest Academic Rejection Letter": _Joke(
        "What would an honest rejection letter say?",
        "Is it because reviewers are constructive?",
        "No — \"Interesting work, but not for us, try again after we've forgotten your name.\"",
    ),
    "Double the monitor, double the fun": _Joke(
        "Does a second monitor double the fun?",
        "Is it because productivity scales linearly?",
        "No — one screen for writing, one for panic-googling, same thesis speed.",
    ),
    "Parental Expectations vs. Time": _Joke(
        "Parental expectations versus time in grad school — who blinks first?",
        "Is it because parents are endlessly patient?",
        "No — they expected a doctor by now; you clarify you're not that kind of doctor.",
    ),
    "Research Diagram/Research Reality": _Joke(
        "Research diagram versus research reality — how do they compare?",
        "Is it because the diagram is simplified?",
        "No — the diagram is clean arrows; reality is a spaghetti plot of failed controls.",
    ),
    "Official Guidelines": _Joke(
        "What do official graduate guidelines promise?",
        "Is it because they map a clear path?",
        "No — a beautiful flowchart that every student navigates like a maze anyway.",
    ),
    "Net Effect of Vacation": _Joke(
        "What is the net effect of vacation on your project?",
        "Is it because rest improves productivity?",
        "No — you return refreshed and behind, with 200 emails and one new committee request.",
    ),
    "Beware the Profzi Scheme": _Joke(
        "What is a profzi scheme?",
        "Is it because it's illegal?",
        "No — your PI promises you'll finish soon if you just run one more experiment for the grant.",
    ),
    "Your Research Interests": _Joke(
        "How specific are your research interests on paper?",
        "Is it because you have a focused niche?",
        "No — broad enough to get funded, narrow enough to panic about job ads.",
    ),
    "Planning": _Joke(
        "How much does planning help in grad school?",
        "Is it because a good plan saves time?",
        "No — the plan is fiction; the Gantt chart is fan fiction.",
    ),
    "Severe Weather Conditions": _Joke(
        "What happens to the lab during severe weather?",
        "Is it because everyone stays home safe?",
        "No — campus closes, but your cells don't, so someone heroically goes in anyway.",
    ),
    "Our work is like this donut": _Joke(
        "Why compare your work to a donut?",
        "Is it because it's sweet and complete?",
        "No — there's a hole in the middle where the data should be.",
    ),
    "Quantum Gradnamics, pt. 2 of 3": _Joke(
        "What is quantum gradnamics?",
        "Is it because you master advanced physics?",
        "No — you're simultaneously done and not done with your thesis until someone observes your PDF.",
    ),
    "So wrong": _Joke(
        "Your advisor says the draft is \"so wrong\" — what now?",
        "Is it because one paragraph needs fixing?",
        "No — you question your entire life while they suggest starting the introduction over.",
    ),
    "The mid-tenure crisis": _Joke(
        "What is the mid-tenure crisis?",
        "Is it because faculty have it easy?",
        "No — they've got grants, papers, and a panic that looks just like yours but with better furniture.",
    ),
    "Parenting: Almost totally worth it": _Joke(
        "Is parenting during grad school almost totally worth it?",
        "Is it because balance is achievable?",
        "No — almost totally worth it, except the days when daycare and defense prep collide.",
    ),
    "C-C-Coffee...": _Joke(
        "Why does the grad student stutter \"C-C-Coffee\"?",
        "Why?",
        "Because it's the only variable holding the experiment — and you — together.",
    ),
    "Dress Codes": _Joke(
        "What is the dress code for defending your thesis?",
        "Is it because you wear formal academic regalia?",
        "No — business on top for Zoom, sweatpants below, panic everywhere.",
    ),
    "How to turn your CV into a Resume": _Joke(
        "How do you turn your CV into a resume?",
        "Is it because you highlight transferable skills?",
        "No — delete eleven pages of presentations and pray \"data analysis\" counts in industry.",
    ),
    "Ifs, buts or maybes": _Joke(
        "How many ifs, buts, or maybes are in your discussion section?",
        "Is it because uncertainty is honest?",
        "No — enough hedging to survive peer review and family dinner questions.",
    ),
    "Cheapest way": _Joke(
        "What's the cheapest way to do this experiment?",
        "Is it because frugal science is good science?",
        "No — reuse tips, borrow reagents, and label it \"pilot study\" on the grant.",
    ),
    "Grad Student Pick up lines": _Joke(
        "What's a grad student pickup line?",
        "Is it because charm wins hearts?",
        "No — \"Are you a statistically significant result? Because you're making my heart p < 0.05.\"",
    ),
    "Keeping your paper within the page limit": _Joke(
        "How do you keep your paper within the page limit?",
        "Is it because you write concisely?",
        "No — shrink the figures, shrink the margins, shrink the methods, shrink your soul.",
    ),
    "The Internet": _Joke(
        "How does the internet help grad students?",
        "Is it because it accelerates research?",
        "No — infinite access to papers you skim while avoiding the one you're supposed to write.",
    ),
    "More of the same": _Joke(
        "Why does every week feel like more of the same?",
        "Is it because routine builds mastery?",
        "No — same experiment, same error message, same vow to finish early next time.",
    ),
    "Cafeteria Potential Well": _Joke(
        "What is the cafeteria potential well?",
        "Is it because the food is surprisingly good?",
        "No — you fall into the cheapest meal and can't escape until stipend day.",
    ),
    "The 2397th Annual Academic Awards": _Joke(
        "Who wins at the annual academic awards?",
        "Is it because merit is celebrated?",
        "No — the person who showed up and the committee that invented a new category for morale.",
    ),
    "Seen on campus": _Joke(
        "What's the strangest thing seen on campus?",
        "Is it because college is quirky?",
        "No — a grad student carrying a ice bucket, a printer, and the look of someone who missed sleep.",
    ),
    "The Higgs Boson Explained": _Joke(
        "Can you explain the Higgs boson at a party?",
        "Is it because you simplify elegantly?",
        "No — you say it gives mass, then mass-distribute appetizers until someone changes topics.",
    ),
    "Lost": _Joke(
        "Why do grad students feel lost?",
        "Is it because the campus map is bad?",
        "No — because the literature is infinite, the path isn't, and your GPS is a vague advisor email.",
    ),
    "In case of emergency": _Joke(
        "What does the lab emergency plan say?",
        "Is it because safety drills happen often?",
        "No — in case of emergency, finish labeling your samples before evacuating.",
    ),
    "Should you ask a question during Seminar?": _Joke(
        "Should you ask a question during seminar?",
        "Is it because participation shows engagement?",
        "No — only if you're ready for a twenty-minute answer and a stare from the speaker.",
    ),
    "Geeks Anonymous": _Joke(
        "What do you admit at Geeks Anonymous?",
        "Is it because you need social skills?",
        "No — \"Hi, I'm in grad school, and I corrected the professor's slide in my head.\"",
    ),
    "Wishful Thinking": _Joke(
        "What is wishful thinking in the lab?",
        "Is it because optimism drives discovery?",
        "No — staring at the data hoping the outlier will confess it was a typo.",
    ),
    "To PhD or not to PhD...": _Joke(
        "To PhD or not to PhD — that is the question?",
        "Is it because Hamlet had options?",
        "No — you chose yes before knowing what \"comprehensive exam\" meant.",
    ),
    "The Lab/Office Fridge": _Joke(
        "What's inside the lab fridge?",
        "Is it because meal prep saves money?",
        "No — samples, lunch, and something brown that predates your enrollment.",
    ),
    "A Grammatical Conundrum": _Joke(
        "What's the grammatical conundrum in your abstract?",
        "Is it because English is hard?",
        "No — passive voice so aggressive the experiment happened to someone unspecified.",
    ),
    "The Night Shift": _Joke(
        "Who works the night shift in the lab?",
        "Is it because science never sleeps?",
        "No — whoever's instrument booking spilled past midnight and called it \"flexibility.\"",
    ),
    "If only": _Joke(
        "You say \"if only\" about your project — if only what?",
        "Is it because one tweak fixes everything?",
        "No — if only you'd started writing in year one instead of year five.",
    ),
    "Fifty Fifty Chance": _Joke(
        "What's the fifty-fifty chance your experiment works?",
        "Is it because you've controlled all variables?",
        "No — fifty percent it fails, fifty percent it fails differently.",
    ),
    "Junk": _Joke(
        "Why is the spare room full of junk?",
        "Is it because labs hoard equipment?",
        "No — every piece is \"might need later\" until it becomes archaeological.",
    ),
    "PhD Propaganda": _Joke(
        "What is PhD propaganda?",
        "Is it because recruitment is honest?",
        "No — brochures with smiling students and no mention of stipend taxes or job odds.",
    ),
    "Your Profile Picture": _Joke(
        "What should your academic profile picture look like?",
        "Is it because professionalism matters?",
        "No — cropped from a conference photo where you were mid-blink but the banner looked official.",
    ),
    "Futile Attempts": _Joke(
        "Why are your attempts to finish early called futile?",
        "Is it because persistence pays off?",
        "No — every \"this semester for sure\" ends the same: extension form.",
    ),
    "Grad SPAM": _Joke(
        "What is grad SPAM?",
        "Is it because email filters fail?",
        "No — listservs selling journals, conferences, and hope you don't read the fine print.",
    ),
    "The Grad Student Brain": _Joke(
        "What does the grad student brain look like under MRI?",
        "Is it because you're cognitively elite?",
        "No — lit review on one side, free food radar on the other, thesis somewhere in the noise.",
    ),
}


def _comparison_joke(title: str) -> _Joke | None:
    m = re.match(r"(.+?)\s+(?:v\.|vs\.?)\s+(.+)", title, flags=re.IGNORECASE)
    if not m:
        return None
    left, right = m.group(1).strip(), m.group(2).strip()
    left_l, right_l = left.lower(), right.lower()
    topic = _topic_phrase(title)
    return _Joke(
        f"In grad school, how does {left_l} stack up against {right_l}?",
        "Why?",
        f"Because {topic} is less a fair fight and more a long-term occupation — and {right_l} usually wins.",
    )


def _explained_joke(title: str) -> _Joke | None:
    if not re.search(r"\bexplained\b", title, re.IGNORECASE):
        return None
    subject = re.sub(r"\s+explained$", "", title, flags=re.IGNORECASE).strip()
    subj_l = subject.lower()
    return _Joke(
        f"Can you explain {subj_l} to someone outside your field?",
        "Is it because it's simple once you draw a diagram?",
        f"No — {subj_l} makes sense in the comic and still hurts when you live it in lab.",
    )


def _how_joke(title: str) -> _Joke | None:
    m = re.match(r"^how\s+(.+)$", title, re.IGNORECASE)
    if not m:
        return None
    rest = m.group(1).rstrip("?").strip().lower()
    return _Joke(
        f"How does {rest} — really?",
        "Why?",
        f"Because asking \"how\" is easy; doing it between funding cycles is the punchline.",
    )


def _why_joke(title: str) -> _Joke | None:
    if not title.lower().startswith("why"):
        return None
    q = title if title.endswith("?") else f"{title}?"
    return _Joke(
        q,
        "Why?",
        "Because grad school turns every simple question into a multi-year longitudinal study.",
    )


def _what_joke(title: str) -> _Joke | None:
    m = re.match(r"^what\s+(.+)$", title, re.IGNORECASE)
    if not m:
        return None
    rest = m.group(1).rstrip("?").strip().lower()
    return _Joke(
        f"What is {rest} — in grad school terms?",
        "Is it because the handbook defines it clearly?",
        f"No — {rest} is whatever your committee says it is this semester.",
    )


def _when_joke(title: str) -> _Joke | None:
    m = re.match(r"^when\s+(.+)$", title, re.IGNORECASE)
    if not m:
        return None
    rest = m.group(1).rstrip("?").strip().lower()
    return _Joke(
        f"When should you {rest}?",
        "Is it because timing is everything?",
        "No — when they're about to leave for the day and can't assign follow-up work.",
    )


def _should_joke(title: str) -> _Joke | None:
    m = re.match(r"^should\s+(.+)$", title, re.IGNORECASE)
    if not m:
        return None
    rest = m.group(1).rstrip("?").strip().lower()
    return _Joke(
        f"Should you {rest}?",
        "Is it because participation shows you're engaged?",
        "No — only if you're ready for consequences that arrive by email.",
    )


def _graph_joke(title: str) -> _Joke | None:
    m = re.match(r"^graph\s*-\s*(.+)$", title, re.IGNORECASE)
    if not m:
        return None
    metric = m.group(1).strip().lower()
    return _Joke(
        f"What does the grad-school graph of {metric} look like?",
        "Is it because it's a smooth upward trend?",
        f"No — flat until deadline panic, then a vertical line that would never pass peer review.",
    )


def _generic_joke(title: str) -> _Joke:
    topic = _topic_phrase(title)
    short = topic if len(topic) < 60 else topic[:57] + "..."
    return _Joke(
        f"What's the grad-school truth about {short}?",
        "Is it because nobody warned you at orientation?",
        f"No — you learn it the first time {short} shows up in lab, email, or a committee meeting.",
    )


def build_joke(title: str) -> str:
    """Return three-line setup/punchline joke text for ``output``."""
    key = _normalize_title(title)
    joke = _JOKES.get(key)
    if joke is None:
        for builder in (
            _graph_joke,
            _comparison_joke,
            _explained_joke,
            _why_joke,
            _when_joke,
            _should_joke,
            _how_joke,
            _what_joke,
        ):
            joke = builder(key)
            if joke is not None:
                break
    if joke is None:
        joke = _generic_joke(key)
    return _fmt(joke)


def build_output_text(record: dict) -> str:
    """Build training ``output`` — never the fake journal abstract."""
    title = record.get("page_title") or record.get("title") or ""
    if not title.strip():
        alt = (record.get("image_alt") or "").strip()
        if alt:
            title = alt
    return build_joke(title)


def to_training_row(record: dict) -> dict[str, str]:
    title = record.get("page_title") or record.get("title") or record.get("image_alt") or ""
    return {
        "instruction": INSTRUCTION,
        "input": build_input(title),
        "output": build_output_text(record),
    }
