"""
Ground truth cases for retrieval evaluation.

Each case maps a query (as the agent would issue it) to the set of chunk IDs
that contain the answer. These were identified by direct OpenSearch keyword
searches — not by the retrieval pipeline itself — so they are independent
ground truth.

Cases are spread across committees and topics to avoid over-fitting to
one political domain. Filter types are varied to prepare for full corpus
ingestion (all available Knesset data).

Fields:
- id: unique case identifier
- query: Hebrew search query (as the agent's planner would generate)
- relevant_chunk_ids: set of chunk _id values that contain the answer
- filters: optional metadata filters the agent should ideally provide
- tags: categories for grouping results
- description: what this case tests
"""

# Committee ID reference (Knesset 25):
# 4186 = Finance, 4187 = National Security, 4189 = Health,
# 4190 = Foreign Affairs & Defense, 4191 = Constitution,
# 4192 = Education, 4193 = Economics, 4194 = Knesset Committee,
# 4196 = Labor & Welfare, 4197 = Immigration, 4198 = Interior,
# 4199 = State Audit, 4208 = Foreign Workers,
# 4266 = AI & Advanced Tech Subcommittee

RETRIEVAL_CASES = [
    # ===== DATE-ONLY FILTERS =====

    # --- Communications / media reform (3 cases) ---
    {
        "id": "karhi_smoking_gun",
        "description": "Find Karhi's 'intent of the intender' quote — hardest needle",
        "query": "כוונת המכוון קרעי רפורמה תקשורת",
        "relevant_chunk_ids": {"2224193_69", "2224193_68"},
        "filters": {
            "from_date": "2024-12-01",
            "to_date": "2024-12-31",
        },
        "tags": ["needle", "communications", "date-filter"],
    },
    {
        "id": "elharar_free_press",
        "description": "Find Elharar's speech about free press and Karhi",
        "query": "קארין אלהרר תקשורת חופשית קרעי",
        "relevant_chunk_ids": {"4812385_66"},
        "filters": {
            "from_date": "2024-11-01",
            "to_date": "2024-12-31",
        },
        "tags": ["needle", "communications", "date-filter"],
    },
    {
        "id": "hostage_deal_anniversary",
        "description": "Find discussion marking one year since first hostage deal",
        "query": "עסקת חטופים שנה ראשונה נתניהו",
        "relevant_chunk_ids": {"4727573_8"},
        "filters": {
            "from_date": "2024-11-01",
            "to_date": "2024-12-31",
        },
        "tags": ["needle", "defense", "date-filter"],
    },

    # ===== COMMITTEE-ONLY FILTERS =====

    {
        "id": "cannabis_ptsd_committee",
        "description": "Find cannabis PTSD discussion filtered by Health Committee only",
        "query": "קנאביס רפואי פוסט טראומה מטופלים",
        "relevant_chunk_ids": {"4805390_88", "4805390_87"},
        "filters": {
            "committee_id": 4189,
        },
        "tags": ["needle", "health", "committee-filter"],
    },
    {
        "id": "gafni_deficit_committee",
        "description": "Gafni on deficit filtered by Finance Committee only",
        "query": "גפני גירעון תקציב תקרה",
        "relevant_chunk_ids": {"5105522_49", "5105522_48"},
        "filters": {
            "committee_id": 4186,
        },
        "tags": ["needle", "finance", "committee-filter"],
    },
    {
        "id": "comptroller_oct7_committee",
        "description": "State comptroller on Oct 7 filtered by Audit Committee only",
        "query": "מבקר המדינה כישלון מערכתי 7 באוקטובר",
        "relevant_chunk_ids": {"5476193_17", "4674502_0"},
        "filters": {
            "committee_id": 4199,
        },
        "tags": ["needle", "audit", "committee-filter"],
    },
    {
        "id": "ai_strategy_committee",
        "description": "AI education strategy filtered by Science & Tech Committee only",
        "query": "בינה מלאכותית חינוך תוכנית אסטרטגית הטמעה",
        "relevant_chunk_ids": {"5438769_111"},
        "filters": {
            "committee_id": 4195,
        },
        "tags": ["needle", "technology", "committee-filter"],
    },

    # ===== COMBINED FILTERS (committee + date) =====

    {
        "id": "al_jazeera_combined",
        "description": "Al Jazeera in Security Committee + Q4 2024",
        "query": "אל ג'זירה שידור הסתה",
        "relevant_chunk_ids": {"4677770_49", "4677770_50", "4677770_47", "4677770_48"},
        "filters": {
            "committee_id": 4187,
            "from_date": "2024-10-01",
            "to_date": "2024-12-31",
        },
        "tags": ["needle", "security", "combined-filter"],
    },
    {
        "id": "bitan_special_combined",
        "description": "Bitan special committee in Economics + Q1 2025",
        "query": "ביטן ועדה מיוחדת תקשורת",
        "relevant_chunk_ids": {"5013005_169", "5013005_170", "5013005_168"},
        "filters": {
            "committee_id": 4193,
            "from_date": "2025-01-01",
            "to_date": "2025-03-31",
        },
        "tags": ["needle", "communications", "combined-filter"],
    },
    {
        "id": "credit_rating_combined",
        "description": "Credit rating in Finance Committee + Nov 2024",
        "query": "דירוג אשראי ישראל הורדה שווקים",
        "relevant_chunk_ids": {"4695981_40", "4695981_0"},
        "filters": {
            "committee_id": 4186,
            "from_date": "2024-11-01",
            "to_date": "2024-12-31",
        },
        "tags": ["needle", "finance", "combined-filter"],
    },
    {
        "id": "absorption_combined",
        "description": "Absorption basket cuts in Immigration Committee + Q4 2024",
        "query": "סל קליטה דיפרנציאלי קיצוץ עולים",
        "relevant_chunk_ids": {"4687011_37", "4687011_35"},
        "filters": {
            "committee_id": 4197,
            "from_date": "2024-10-01",
            "to_date": "2024-12-31",
        },
        "tags": ["needle", "immigration", "combined-filter"],
    },
    {
        "id": "foreign_workers_combined",
        "description": "Foreign workers in Gaza envelope + Foreign Workers Committee + Q4-Q1",
        "query": "עובדים זרים חקלאות עוטף ��זה שיקום",
        "relevant_chunk_ids": {"5033450_45"},
        "filters": {
            "committee_id": 4208,
            "from_date": "2024-12-01",
            "to_date": "2025-03-31",
        },
        "tags": ["needle", "labor", "combined-filter"],
    },

    # ===== BROAD / CROSS-COMMITTEE (no committee filter) =====

    {
        "id": "press_freedom_broad",
        "description": "Press freedom across all committees",
        "query": "חופש העיתונות ביטחון שידור",
        "relevant_chunk_ids": {"4677770_47", "4677770_48"},
        "filters": {
            "from_date": "2024-10-01",
            "to_date": "2024-12-31",
        },
        "tags": ["broad", "security", "date-filter"],
    },

    # ===== NO-FILTER STRESS TESTS =====

    {
        "id": "no_filter_karhi",
        "description": "Karhi smoking gun WITHOUT any filters — stress test",
        "query": "קרעי רפורמה תקשורת ועדת החוקה",
        "relevant_chunk_ids": {"2224193_69", "2224193_68"},
        "filters": {},
        "tags": ["no-filter", "stress-test"],
    },
    {
        "id": "no_filter_gafni",
        "description": "Gafni deficit WITHOUT any filters — stress test",
        "query": "גפני גירעון תקציב תקרה",
        "relevant_chunk_ids": {"5105522_49", "5105522_48"},
        "filters": {},
        "tags": ["no-filter", "stress-test"],
    },
    {
        "id": "no_filter_cannabis",
        "description": "Cannabis PTSD WITHOUT any filters — stress test",
        "query": "קנאבי�� רפואי פוסט טראומה מטופלים",
        "relevant_chunk_ids": {"4805390_88", "4805390_87"},
        "filters": {},
        "tags": ["no-filter", "stress-test"],
    },
]
