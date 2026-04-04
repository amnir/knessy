"""
Benchmark test cases for the Knesset research agent.

Each case defines:
- question: The Hebrew/English question to ask the agent
- expected: List of evidence markers that SHOULD appear in the answer.
            Each marker is a dict with:
              "text": substring to search for (Hebrew or English)
              "weight": importance (1=nice-to-have, 2=important, 3=critical)
              "description": what this marker represents (for reporting)
- negative: List of substrings that should NOT appear (hallucination checks)
- tags: Categories for filtering (e.g. "retrieval", "synthesis", "cross-committee")
"""

BENCHMARK_CASES = [
    {
        "id": "karhi_political_motivation",
        "question": "מה אמר שר התקשורת שלמה קרעי בדיוני הוועדות על המניעים לחוק התקשורת? האם הוא הודה במניע פוליטי?",
        "expected": [
            {
                "text": "קרעי",
                "weight": 3,
                "description": "Mentions Karhi by name",
            },
            {
                "text": "כוונת המכוון",
                "weight": 3,
                "description": "Key quote: 'the intent' — the smoking gun phrase from Constitution Committee",
            },
            {
                "text": "רפורמה",
                "weight": 2,
                "description": "References the reform",
            },
            {
                "text": "ועדת החוקה",
                "weight": 2,
                "description": "Identifies Constitution Committee as source",
            },
            {
                "text": "2024-12",
                "weight": 2,
                "description": "Correct date range (December 2024)",
            },
            {
                "text": "חוק פרטי",
                "weight": 1,
                "description": "Mentions the reform was split into private bills",
            },
        ],
        "negative": [
            "הודה במניע פוליטי",  # Karhi didn't directly admit — agent shouldn't claim he did
        ],
        "tags": ["retrieval", "needle-in-haystack"],
    },
    {
        "id": "communications_oversight_2024",
        "question": "נתח את הפיקוח הפרלמנטרי של ועדות הכנסת על מדיניות התקשורת במהלך 2024. אילו ועדות עסקו בנושא, מה היו הדיונים המרכזיים, והאם הועלו חששות לגבי תהליכי החקיקה?",
        "expected": [
            {
                "text": "ביטחון לאומי",
                "weight": 3,
                "description": "Identifies National Security Committee",
            },
            {
                "text": "כלכלה",
                "weight": 2,
                "description": "Identifies Economics Committee",
            },
            {
                "text": "חופש העיתונות",
                "weight": 2,
                "description": "Mentions press freedom concerns",
            },
            {
                "text": "Session",
                "weight": 3,
                "description": "Cites at least one specific session ID",
            },
            {
                "text": "2024",
                "weight": 2,
                "description": "Stays within the 2024 timeframe",
            },
        ],
        "negative": [],
        "tags": ["synthesis", "cross-committee", "open-analysis"],
    },
    {
        "id": "elharar_opposition",
        "question": "מה אמרה חברת הכנסת קארין אלהרר על הצעות החוק של שר התקשורת קרעי?",
        "expected": [
            {
                "text": "אלהרר",
                "weight": 3,
                "description": "Mentions Elharar by name",
            },
            {
                "text": "קרעי",
                "weight": 3,
                "description": "Connects to Karhi",
            },
            {
                "text": "דמוקרטית",
                "weight": 2,
                "description": "References democracy argument",
            },
            {
                "text": "חופשי",
                "weight": 2,
                "description": "References free press argument",
            },
            {
                "text": "ועדת החוקה",
                "weight": 2,
                "description": "Identifies Constitution Committee as source",
            },
            {
                "text": "2024-11",
                "weight": 1,
                "description": "Correct date (November 2024)",
            },
        ],
        "negative": [],
        "tags": ["retrieval", "specific-quote"],
    },
    {
        "id": "al_jazeera_ban",
        "question": "What was discussed in Knesset committees regarding banning Al Jazeera from broadcasting in Israel?",
        "expected": [
            {
                "text": "ג'זירה",
                "weight": 3,
                "description": "Mentions Al Jazeera in Hebrew",
            },
            {
                "text": "ביטחון לאומי",
                "weight": 2,
                "description": "Identifies National Security Committee",
            },
            {
                "text": "Session",
                "weight": 2,
                "description": "Cites specific session",
            },
        ],
        "negative": [],
        "tags": ["retrieval", "english-query"],
    },
    {
        "id": "special_communications_committee",
        "question": "כיצד הוקמה הוועדה המיוחדת לתקשורת בכנסת ה-25? מדוע לא דנה ועדת הכלכלה בחוק? מה היו הטענות בעד ונגד?",
        "expected": [
            {
                "text": "ביטן",
                "weight": 3,
                "description": "Mentions Economics Committee chair David Bitan",
            },
            {
                "text": "המיוחדת",
                "weight": 3,
                "description": "References the special committee",
            },
            {
                "text": "ועדת הכלכלה",
                "weight": 2,
                "description": "Identifies Economics Committee as the bypassed committee",
            },
            {
                "text": "קרעי",
                "weight": 2,
                "description": "Connects to Minister Karhi",
            },
            {
                "text": "Session",
                "weight": 2,
                "description": "Cites at least one specific session",
            },
            {
                "text": "2025",
                "weight": 1,
                "description": "Correct timeframe (early 2025)",
            },
        ],
        "negative": [],
        "tags": ["synthesis", "cross-committee", "committee-bypass"],
    },
]
