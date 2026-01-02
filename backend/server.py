from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from langdetect import detect

app = Flask(__name__)
CORS(app)

# --- 1. MODEL CONFIGURATION ---
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"SYSTEM STARTUP: Loading AI Model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
print("Model Loaded Successfully!")

# --- 2. LANGUAGE MAPPING ---
LANG_CODE_MAP = {
    "en": "eng_Latn", "ur": "urd_Arab", 
    "ar": "ara_Arab", "fr": "fra_Latn"
}

# --- 3. RULE-BASED GLOSSARY ---
URDU_GLOSSARY = {
 
    "butterflies in my stomach": {
        "bad": ["میرے پیٹ میں تتلیاں", "پیٹ میں تتلی"],
        "good": "دل گھبرانا / خوفزدہ ہونا"
    },
    "over the moon": {
        "bad": ["چاند کے اوپر", "چاند پر"],
        "good": "بے حد خوش / پھولے نہ سمانا"
    },
    "on cloud nine": {
        "bad": ["بادل نو پر", "نویں بادل پر"],
        "good": "ساتویں آسمان پر / بہت خوش"
    },
    "under the weather": {
        "bad": ["موسم کے تحت", "موسم کے نیچے"],
        "good": "طبیعت ناساز"
    },
    "heart of gold": {
        "bad": ["سونے کا دل"],
        "good": "انتہائی نیک دل / رحم دل"
    },
    "cold feet": {
        "bad": ["ٹھنڈے پاؤں", "پاؤں ٹھنڈے"],
        "good": "ہمت ہار جانا / گھبرا جانا"
    },
    "piece of cake": {
        "bad": ["کیک کا ایک ٹکڑا", "کیک کا ٹکڑا", "کیک کا حصہ"],
        "good": "بائیں ہاتھ کا کھیل / بہت آسان کام"
    },
    "break a leg": {
        "bad": ["ٹانگ توڑ دو", "ٹانگ توڑنا"],
        "good": "اللہ کامیاب کرے / گڈ لک"
    },
    "hit the sack": {
        "bad": ["بوری کو مارنا", "بوری مارو"],
        "good": "سو جانا / بستر پر جانا"
    },
    "burn the midnight oil": {
        "bad": ["آدھی رات کا تیل جلانا", "رات کا تیل جلائیں"],
        "good": "رات دن ایک کرنا / سخت محنت کرنا"
    },
    "call it a day": {
        "bad": ["اسے ایک دن کہو", "اسے دن بلاؤ"],
        "good": "آج کے لیے کام ختم کرنا"
    },
    "cut corners": {
        "bad": ["کونے کاٹنا", "کونے کاٹ دیں"],
        "good": "کام میں ڈنڈی مارنا / بچت کے لیے معیار گرانا"
    },
    "miss the boat": {
        "bad": ["کشتی چھوٹ جانا", "کشتی کو یاد کرنا"],
        "good": "موقع گنوا دینا"
    },
    "beat around the bush": {
        "bad": ["جھاڑی کے گرد مارنا", "جھاڑی کو پیٹنا"],
        "good": "ادھر ادھر کی باتیں کرنا / اصل مدعے پر نہ آنا"
    },
    "hit the nail on the head": {
        "bad": ["سر پر کیل مارنا", "کیل کو سر پر مارو"],
        "good": "بالکل درست بات کرنا / نشانہ پر لگنا"
    },
    "spill the beans": {
        "bad": ["پھلیاں گرانا", "پھلیاں بہا دو"],
        "good": "راز فاش کر دینا / بھانڈا پھوڑنا"
    },
    "bite the bullet": {
        "bad": ["گولی کاٹنا", "گولی کاٹو"],
        "good": "کڑوا گھونٹ بھرنا / مشکل برداشت کرنا"
    },
    "apple of my eye": {
        "bad": ["میری آنکھ کا سیب", "آنکھ کا سیب"],
        "good": "آنکھ کا تارہ / بہت عزیز"
    },
    "once in a blue moon": {
        "bad": ["نیلے چاند میں ایک بار", "نیلے چاند میں"],
        "good": "کبھی کبھار / عید کا چاند"
    },
    "add fuel to the fire": {
        "bad": ["آگ میں ایندھن ڈالنا", "آگ پر تیل"],
        "good": "جلتی پر تیل ڈالنا"
    },
    "blessing in disguise": {
        "bad": ["بھیس میں نعمت", "چھپی ہوئی نعمت"],
        "good": "زحمت میں رحمت"
    },
    "cost an arm and a leg": {
        "bad": ["ایک بازو اور ٹانگ کی لاگت", "ہاتھ اور ٹانگ کی قیمت"],
        "good": "بہت مہنگا پڑنا / بھاری قیمت چکانا"
    },
    "cry over spilt milk": {
        "bad": ["گرے ہوئے دودھ پر رونا", "بکھرے دودھ پر رونا"],
        "good": "اب پچھتائے کیا ہوت جب چڑیاں چگ گئیں کھیت"
    },
    "actions speak louder than words": {
        "bad": ["عمل الفاظ سے زیادہ بلند بولتے ہیں"],
        "good": "عمل کا اثر باتوں سے زیادہ ہوتا ہے"
    },
    "barking up the wrong tree": {
        "bad": ["غلط درخت پر بھونکنا"],
        "good": "غلط فہمی کا شکار ہونا / غلط جگہ کوشش کرنا"
    },
    "kill two birds with one stone": {
        "bad": ["ایک پتھر سے دو پرندے مارنا"],
        "good": "ایک تیر سے دو شکار"
    },
    "better late than never": {
        "bad": ["کبھی نہیں سے دیر بہتر"],
        "good": "دیر آید درست آید"
    },
    "face the music": {
        "bad": ["موسیقی کا سامنا", "میوزک کا سامنا"],
        "good": "نتائج بھگتنا / کیے کی سزا پانا"
    },
    "through thick and thin": {
        "bad": ["موٹے اور پتلے کے ذریعے"],
        "good": "اچھے برے حالات میں / سکھ دکھ میں"
    },
    "turn a blind eye": {
        "bad": ["اندھی آنکھ موڑنا", "آنکھیں بند کرنا"],
        "good": "نظر انداز کرنا / جان بوجھ کر انجان بننا"
    },
    "wild goose chase": {
        "bad": ["جنگلی ہنس کا پیچھا"],
        "good": "لا حاصل کوشش / بے کار بھاگ دوڑ"
    }
}

# --- 4. HELPER FUNCTION ---
def apply_glossary_rules(source_text, translated_text, target_lang):
    """
    Checks source text for idioms and replaces literal translations 
    in the output with cultural equivalents.
    """
    # Currently only applying rules for Urdu
    if target_lang != "ur":
        return translated_text

    lower_source = source_text.lower()

    # Iterate through our dictionary
    for idiom, rules in URDU_GLOSSARY.items():
        if idiom in lower_source:
            # Check if any of the "bad" literal translations appear in the output
            for bad_translation in rules["bad"]:
                if bad_translation in translated_text:
                    # Replace with the "good" meaning
                    translated_text = translated_text.replace(bad_translation, rules["good"])
    
    return translated_text

# --- 5. API ENDPOINT ---
@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text', '')
    target_lang_key = data.get('target_lang', 'en')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Detect & Map Languages
        try:
            detected_lang = detect(text)
            if detected_lang not in LANG_CODE_MAP: detected_lang = "en"
        except:
            detected_lang = "en"

        source_code = LANG_CODE_MAP.get(detected_lang, "eng_Latn")
        target_code = LANG_CODE_MAP.get(target_lang_key, "eng_Latn")

        # Tokenize
        tokenizer.src_lang = source_code
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # Fix for NLLB Tokenizer (using convert_tokens_to_ids)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_code)

        # Generate
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=200, 
                num_beams=5,
                early_stopping=True
            )

        translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        # --- APPLY GLOSSARY RULES HERE ---
        # This runs AFTER the AI generates the text but BEFORE sending it to user
        final_text = apply_glossary_rules(text, translated_text, target_lang_key)

        return jsonify({
            "original_text": text,
            "detected_language": detected_lang,
            "translation": final_text
        })

    except Exception as e:
        print(f"SERVER ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)