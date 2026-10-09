"""
advice.py — Treatment and prevention advice for CropDoctor
===========================================================
Provides plant-care guidance for each supported disease class.

Guidance philosophy
-------------------
1. Low-cost, non-chemical options are listed first.
2. Chemical treatments are mentioned only in general terms.
   Specific dosage rates or spray schedules are NOT provided here — always
   follow the product label and your local agricultural authority's guidelines.
3. When in doubt, or if symptoms persist, consult a qualified agronomist or
   your nearest agriculture extension officer.

Optional LLM integration
------------------------
Set the following environment variables in a .env file (never commit this):

  # Option A — IBM watsonx.ai
  LLM_PROVIDER=watsonx
  WATSONX_API_KEY=<your key>
  WATSONX_PROJECT_ID=<your project id>
  WATSONX_URL=https://us-south.ml.cloud.ibm.com

  # Option B — OpenAI-compatible endpoint
  LLM_PROVIDER=openai
  OPENAI_API_KEY=<your key>
  OPENAI_MODEL=gpt-3.5-turbo        # optional, default shown

If LLM_PROVIDER is not set or the API call fails, the app falls back to the
local advice dictionary below — no internet connection required.
"""

import os
from pathlib import Path
from typing import Optional

# Load .env if present (safe no-op if python-dotenv is not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Local advice dictionary
# ---------------------------------------------------------------------------
# Each key matches a class name from dataset.DEFAULT_CLASSES.
# Advice is split into sections for readability in the Streamlit UI.

ADVICE: dict = {
    # ------------------------------------------------------------------
    # TOMATO
    # ------------------------------------------------------------------
    "Tomato_Early_blight": {
        "disease_name": "Tomato Early Blight (Alternaria solani)",
        "symptoms": (
            "Dark-brown spots with concentric rings (target-board pattern) on "
            "older leaves.  Leaves may yellow and drop prematurely."
        ),
        "cultural_controls": [
            "Remove and destroy heavily infected leaves to reduce spore load.",
            "Avoid overhead watering; water at the base in the morning so "
            "foliage dries quickly.",
            "Rotate crops — do not plant tomatoes in the same bed for at least "
            "2–3 seasons.",
            "Space plants to improve air circulation.",
            "Mulch the soil surface to prevent soil-splash onto lower leaves.",
            "Disinfect pruning tools between plants (diluted bleach or alcohol).",
        ],
        "chemical_guidance": (
            "Fungicide applications (e.g., copper-based or chlorothalonil products) "
            "can reduce spread.  Follow the registered product label for rates and "
            "intervals, and consult your local agricultural officer for locally "
            "approved products."
        ),
        "when_to_seek_help": (
            "If more than 30 % of the canopy is affected, or if the problem "
            "recurs each season, contact an agriculture extension officer for a "
            "site-specific management plan."
        ),
    },

    "Tomato_Late_blight": {
        "disease_name": "Tomato Late Blight (Phytophthora infestans)",
        "symptoms": (
            "Water-soaked, dark-brown or black irregular lesions on leaves, stems, "
            "and fruit.  A white mouldy growth may appear on the underside of "
            "leaves in humid weather."
        ),
        "cultural_controls": [
            "Remove and bag (do not compost) infected plant material immediately.",
            "Avoid overhead irrigation; keep foliage dry.",
            "Improve drainage and air circulation around plants.",
            "Avoid working among plants when foliage is wet.",
            "Plant resistant varieties when available.",
            "Do not leave cull tomatoes or volunteer plants in the field.",
        ],
        "chemical_guidance": (
            "Late blight spreads rapidly.  Protective fungicide applications "
            "(copper-based or systemic products) should begin early if conditions "
            "favour infection (cool, wet weather).  Follow the product label and "
            "local guidelines."
        ),
        "when_to_seek_help": (
            "Late blight can destroy an entire crop within days under favourable "
            "weather.  Contact your agriculture extension officer promptly if you "
            "suspect this disease."
        ),
    },

    "Tomato_Leaf_Miner": {
        "disease_name": "Tomato Leaf Miner (Tuta absoluta / Liriomyza spp.)",
        "symptoms": (
            "Winding silvery or white tunnels (mines) visible on the leaf surface, "
            "created by larvae feeding inside leaf tissue."
        ),
        "cultural_controls": [
            "Remove and destroy heavily mined leaves.",
            "Use yellow sticky traps to monitor and reduce adult fly populations.",
            "Introduce or conserve natural enemies (parasitic wasps) where possible.",
            "Use row covers or insect-proof mesh on seedlings.",
            "Remove crop debris promptly after harvest.",
        ],
        "chemical_guidance": (
            "Some insecticides (e.g., abamectin, spinosad) are registered for leaf "
            "miners.  Rotate chemical classes to delay resistance.  Follow the "
            "product label and adhere to pre-harvest intervals."
        ),
        "when_to_seek_help": (
            "If mines appear on the majority of leaves, consult an agronomist "
            "for a resistance management and spray timing strategy."
        ),
    },

    "Tomato_healthy": {
        "disease_name": "Healthy Tomato",
        "symptoms": "No disease symptoms detected.",
        "cultural_controls": [
            "Maintain good watering practices — consistent moisture prevents "
            "blossom-end rot and fruit cracking.",
            "Feed with a balanced fertiliser; avoid excess nitrogen late in season.",
            "Stake or cage plants to keep foliage off the ground.",
            "Scout regularly (at least twice a week) for early signs of pests or "
            "disease so problems can be caught early.",
        ],
        "chemical_guidance": None,
        "when_to_seek_help": (
            "Contact an agronomist if you notice unusual wilting, discolouration, "
            "or unexplained crop losses."
        ),
    },

    # ------------------------------------------------------------------
    # POTATO
    # ------------------------------------------------------------------
    "Potato___Early_blight": {
        "disease_name": "Potato Early Blight (Alternaria solani)",
        "symptoms": (
            "Small, dark-brown spots with a concentric-ring pattern on older, "
            "lower leaves.  Severe infections cause premature defoliation and "
            "reduced tuber yield."
        ),
        "cultural_controls": [
            "Use certified, disease-free seed potatoes.",
            "Remove and destroy infected leaves.",
            "Rotate crops — avoid planting potatoes or tomatoes in the same soil "
            "for 2–3 years.",
            "Improve soil drainage and avoid over-irrigation.",
            "Space plants adequately for good air movement.",
        ],
        "chemical_guidance": (
            "Preventive fungicide applications (copper, mancozeb, or other "
            "registered products) are often applied on a calendar or weather-based "
            "schedule.  Follow label directions and local recommendations."
        ),
        "when_to_seek_help": (
            "Consult an extension officer if the disease develops rapidly or if "
            "you are unsure of the correct fungicide rotation."
        ),
    },

    "Potato___Late_blight": {
        "disease_name": "Potato Late Blight (Phytophthora infestans)",
        "symptoms": (
            "Irregular, water-soaked dark lesions on leaves and stems that enlarge "
            "rapidly.  White sporulation on lesion edges in humid conditions.  "
            "Tubers may develop a brown, granular rot."
        ),
        "cultural_controls": [
            "Plant certified, disease-free seed potatoes.",
            "Destroy volunteer potato plants and cull piles.",
            "Hill soil around the base of plants to protect tubers.",
            "Harvest tubers in dry conditions; cure properly before storage.",
            "Avoid overhead irrigation.",
        ],
        "chemical_guidance": (
            "Fungicides (systemic or protective) applied preventively are the main "
            "chemical tool.  Timing is critical — start before symptoms appear "
            "during high-risk weather.  Rotate fungicide modes of action to manage "
            "resistance.  Follow the label."
        ),
        "when_to_seek_help": (
            "Late blight is a notifiable disease in many regions.  Report suspected "
            "outbreaks to your local agricultural authority and seek immediate "
            "professional advice."
        ),
    },

    "Potato___healthy": {
        "disease_name": "Healthy Potato",
        "symptoms": "No disease symptoms detected.",
        "cultural_controls": [
            "Use certified seed potatoes.",
            "Rotate crops and maintain good soil health with organic matter.",
            "Monitor soil moisture — avoid waterlogging.",
            "Check regularly for early signs of blight, aphids, or Colorado beetle.",
        ],
        "chemical_guidance": None,
        "when_to_seek_help": (
            "Consult an extension officer or agronomist if you observe wilting, "
            "unusual spots, or abnormal tuber development."
        ),
    },

    # ------------------------------------------------------------------
    # PEPPER (BELL)
    # ------------------------------------------------------------------
    "Pepper__bell___Bacterial_spot": {
        "disease_name": "Pepper Bacterial Spot (Xanthomonas campestris pv. vesicatoria)",
        "symptoms": (
            "Small, water-soaked spots on leaves and fruit that turn dark brown "
            "with yellow halos.  Severely infected leaves drop; fruit lesions "
            "are raised and scab-like."
        ),
        "cultural_controls": [
            "Use certified, disease-free seed and transplants.",
            "Avoid working among plants when foliage is wet.",
            "Remove and destroy heavily infected plant parts.",
            "Use drip irrigation rather than overhead sprinklers.",
            "Rotate crops — do not plant peppers or tomatoes in the same area "
            "for at least 2 years.",
            "Disinfect tools and stakes between uses.",
        ],
        "chemical_guidance": (
            "Copper-based bactericides can reduce spread, especially when applied "
            "preventively.  Copper resistance is increasingly common — consult "
            "your local agricultural authority for current recommendations."
        ),
        "when_to_seek_help": (
            "If the disease is severe or spreading rapidly, contact an agronomist "
            "for advice on copper alternatives and resistant variety selection."
        ),
    },

    "Pepper__bell___healthy": {
        "disease_name": "Healthy Bell Pepper",
        "symptoms": "No disease symptoms detected.",
        "cultural_controls": [
            "Maintain consistent soil moisture — irregular watering causes "
            "blossom-end rot.",
            "Provide adequate calcium through balanced fertilisation.",
            "Scout regularly for aphids, whiteflies, and spider mites.",
            "Stake taller varieties to prevent stem breakage.",
        ],
        "chemical_guidance": None,
        "when_to_seek_help": (
            "Seek professional advice if you notice widespread wilting, fruit "
            "drop, or symptoms you cannot identify."
        ),
    },

    # ------------------------------------------------------------------
    # CORN / MAIZE
    # ------------------------------------------------------------------
    "Corn_(maize)___Common_rust_": {
        "disease_name": "Corn Common Rust (Puccinia sorghi)",
        "symptoms": (
            "Small, powdery, brick-red to brown pustules on both leaf surfaces. "
            "Pustules may turn dark (black) as the season progresses."
        ),
        "cultural_controls": [
            "Plant rust-resistant hybrid varieties when available.",
            "Plant early in the season to avoid peak rust periods.",
            "Remove and destroy severely infected leaves to reduce inoculum.",
            "Avoid dense planting that limits air circulation.",
        ],
        "chemical_guidance": (
            "Fungicide applications (triazole or strobilurin products) can be "
            "economically justified when rust appears before tasselling on "
            "susceptible varieties.  Follow label directions and local guidelines."
        ),
        "when_to_seek_help": (
            "Consult an extension officer if you are unsure whether fungicide "
            "application is economically warranted for your variety and yield "
            "potential."
        ),
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "disease_name": "Corn Northern Leaf Blight (Exserohilum turcicum)",
        "symptoms": (
            "Long (5–15 cm), cigar-shaped, grey-green to tan lesions on leaves. "
            "Lesions may have darker borders and visible spore masses in humid "
            "conditions."
        ),
        "cultural_controls": [
            "Plant resistant hybrids — this is the most effective and economical "
            "control measure.",
            "Rotate crops with non-host species (e.g., soybeans, legumes).",
            "Incorporate crop residue after harvest to reduce inoculum.",
            "Avoid dense planting.",
        ],
        "chemical_guidance": (
            "Fungicides can reduce yield loss on susceptible varieties when "
            "applied before or at early infection.  Follow the label and local "
            "extension recommendations for timing and product selection."
        ),
        "when_to_seek_help": (
            "If more than 50 % of the leaf area is affected before grain-fill, "
            "significant yield loss is likely.  Consult an agronomist."
        ),
    },

    "Corn_(maize)___healthy": {
        "disease_name": "Healthy Maize (Corn)",
        "symptoms": "No disease symptoms detected.",
        "cultural_controls": [
            "Maintain balanced soil fertility, especially nitrogen.",
            "Ensure adequate plant spacing for good air circulation.",
            "Scout fields regularly for fall armyworm, stalk borers, and disease.",
            "Rotate crops each season.",
        ],
        "chemical_guidance": None,
        "when_to_seek_help": (
            "Contact an agronomist or extension officer if you observe unusual "
            "stunting, ear problems, or widespread leaf damage."
        ),
    },
}

# ---------------------------------------------------------------------------
# LLM integration (optional, with local fallback)
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").lower().strip()


def _format_local_advice(class_name: str) -> str:
    """Return a formatted multi-line string from the local advice dict."""
    info = ADVICE.get(class_name)
    if info is None:
        return (
            f"No specific advice is available for class '{class_name}'.\n"
            "Please consult your local agriculture extension officer."
        )

    lines = []
    lines.append(f"**{info['disease_name']}**\n")

    if info.get("symptoms"):
        lines.append(f"**Symptoms:**\n{info['symptoms']}\n")

    if info.get("cultural_controls"):
        lines.append("**Cultural / Non-Chemical Controls:**")
        for tip in info["cultural_controls"]:
            lines.append(f"  • {tip}")
        lines.append("")

    if info.get("chemical_guidance"):
        lines.append(f"**Chemical Guidance (General):**\n{info['chemical_guidance']}\n")

    if info.get("when_to_seek_help"):
        lines.append(f"**When to Seek Professional Help:**\n{info['when_to_seek_help']}")

    return "\n".join(lines)


def _get_watsonx_advice(class_name: str) -> Optional[str]:
    """
    Query IBM watsonx.ai for treatment advice.
    Returns None if the call fails or credentials are missing.
    """
    api_key    = os.getenv("WATSONX_API_KEY", "")
    project_id = os.getenv("WATSONX_PROJECT_ID", "")
    url        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    model_id   = os.getenv("WATSONX_MODEL_ID", "ibm/granite-13b-instruct-v2")

    if not api_key or not project_id:
        return None

    try:
        import requests  # local import to keep top-level deps minimal

        # Authenticate
        auth_resp = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": api_key,
            },
            timeout=15,
        )
        auth_resp.raise_for_status()
        access_token = auth_resp.json()["access_token"]

        prompt = (
            f"You are an experienced agronomist helping smallholder farmers in "
            f"developing countries.  A crop disease detection AI identified the "
            f"following condition: '{class_name.replace('_', ' ')}'. "
            f"Provide brief, practical treatment and prevention advice in plain "
            f"language.  List low-cost and non-chemical options first.  "
            f"Do not invent specific pesticide doses.  Recommend consulting a "
            f"local agriculture officer if the diagnosis is uncertain."
        )

        gen_url = f"{url}/ml/v1/text/generation?version=2023-05-29"
        payload = {
            "model_id": model_id,
            "project_id": project_id,
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens":  300,
                "stop_sequences":  [],
            },
        }
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type":  "application/json",
        }
        resp = requests.post(gen_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        generated = resp.json()["results"][0]["generated_text"].strip()
        return generated if generated else None

    except Exception as exc:
        print(f"[advice.py] watsonx call failed: {exc}. Using local advice.")
        return None


def _get_openai_advice(class_name: str) -> Optional[str]:
    """
    Query an OpenAI-compatible endpoint for treatment advice.
    Returns None if the call fails or credentials are missing.
    """
    api_key   = os.getenv("OPENAI_API_KEY", "")
    model     = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    base_url  = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        return None

    try:
        import requests

        prompt = (
            f"You are an experienced agronomist.  A crop disease AI identified "
            f"'{class_name.replace('_', ' ')}' on a farmer's leaf image.  "
            f"Give brief, practical treatment and prevention guidance.  List "
            f"non-chemical options first.  Do not invent pesticide doses or "
            f"schedules.  Advise consulting a local agriculture officer if unsure."
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 350,
        }
        resp = requests.post(
            f"{base_url}/chat/completions",
            json=payload, headers=headers, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    except Exception as exc:
        print(f"[advice.py] OpenAI call failed: {exc}. Using local advice.")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_advice(class_name: str, use_llm: bool = True) -> str:
    """
    Return treatment and prevention advice for a predicted class.

    Parameters
    ----------
    class_name : str
        The class name returned by the predictor.
    use_llm : bool
        If True (default), try the configured LLM first; fall back to local.
        If False, always use the local dictionary.

    Returns
    -------
    str
        Formatted advice text (may contain markdown).
    """
    if use_llm:
        generated = None
        if LLM_PROVIDER == "watsonx":
            generated = _get_watsonx_advice(class_name)
        elif LLM_PROVIDER == "openai":
            generated = _get_openai_advice(class_name)

        if generated:
            note = (
                "\n\n---\n*Advice generated by AI assistant.  Always follow your "
                "local agricultural authority's guidelines and consult a qualified "
                "agronomist when in doubt.*"
            )
            return generated + note

    # Fallback: local dictionary
    return _format_local_advice(class_name)


def get_advice_dict(class_name: str) -> dict:
    """Return the raw advice dict (structured) for a class name, or {}."""
    return ADVICE.get(class_name, {})


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    cls = sys.argv[1] if len(sys.argv) > 1 else "Tomato_Early_blight"
    print(f"\n{'='*60}")
    print(f"  Advice for: {cls}")
    print(f"{'='*60}\n")
    print(get_advice(cls, use_llm=False))
