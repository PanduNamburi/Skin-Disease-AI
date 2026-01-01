"""
Disease Information Database
Contains resources, treatment recommendations, and doctor consultation guidelines
for each skin disease class.
"""

DISEASE_INFO = {
    "Acne": {
        "description": "A common skin condition that occurs when hair follicles become plugged with oil and dead skin cells.",
        "resources": [
            "American Academy of Dermatology: Acne Treatment",
            "Mayo Clinic: Acne Overview",
            "WebMD: Acne Guide"
        ],
        "treatment_recommendations": [
            "Keep skin clean with gentle cleansers",
            "Use non-comedogenic (won't clog pores) skincare products",
            "Avoid picking or squeezing pimples",
            "Consider over-the-counter treatments with benzoyl peroxide or salicylic acid",
            "For severe cases, consult a dermatologist for prescription medications"
        ],
        "consult_doctor": "Consult a dermatologist if: acne is severe, causes emotional distress, doesn't improve with over-the-counter treatments, or leaves scars."
    },
    "Actinic_Keratosis": {
        "description": "Rough, scaly patches on the skin caused by years of sun exposure. Can be a precancerous condition.",
        "resources": [
            "Skin Cancer Foundation: Actinic Keratosis",
            "American Academy of Dermatology: Actinic Keratosis",
            "DermNet NZ: Actinic Keratosis"
        ],
        "treatment_recommendations": [
            "Protect skin from sun exposure with SPF 30+ sunscreen",
            "Wear protective clothing and hats",
            "Avoid tanning beds",
            "Regular skin examinations by a dermatologist",
            "Treatment options include cryotherapy, topical medications, or photodynamic therapy"
        ],
        "consult_doctor": "Always consult a dermatologist immediately. Actinic keratosis can develop into skin cancer and requires professional evaluation and treatment."
    },
    "Benign_tumors": {
        "description": "Non-cancerous growths on the skin that are usually harmless but should be monitored.",
        "resources": [
            "American Academy of Dermatology: Benign Skin Growths",
            "Mayo Clinic: Skin Growths",
            "Healthline: Benign Skin Tumors"
        ],
        "treatment_recommendations": [
            "Monitor for changes in size, color, or shape",
            "Protect from sun exposure",
            "Most benign tumors don't require treatment unless they cause discomfort",
            "Surgical removal may be recommended if they interfere with daily activities"
        ],
        "consult_doctor": "Consult a dermatologist if: the growth changes in appearance, bleeds, itches, or causes concern. Regular check-ups recommended."
    },
    "Bullous": {
        "description": "A group of autoimmune skin disorders characterized by blisters and erosions.",
        "resources": [
            "National Organization for Rare Disorders: Bullous Pemphigoid",
            "DermNet NZ: Bullous Diseases",
            "American Academy of Dermatology: Blistering Diseases"
        ],
        "treatment_recommendations": [
            "Avoid skin trauma and friction",
            "Keep blisters clean and covered",
            "Use gentle, fragrance-free skincare products",
            "Treatment typically requires prescription medications (corticosteroids, immunosuppressants)",
            "Follow dermatologist's treatment plan closely"
        ],
        "consult_doctor": "Consult a dermatologist immediately. Bullous diseases are serious autoimmune conditions requiring medical treatment and monitoring."
    },
    "Candidiasis": {
        "description": "A fungal infection caused by Candida yeast, commonly affecting skin folds and moist areas.",
        "resources": [
            "CDC: Candidiasis Information",
            "Mayo Clinic: Yeast Infection",
            "WebMD: Candidiasis Guide"
        ],
        "treatment_recommendations": [
            "Keep affected areas clean and dry",
            "Use antifungal creams (clotrimazole, miconazole)",
            "Wear loose, breathable clothing",
            "Change out of wet clothing promptly",
            "For persistent cases, oral antifungal medications may be needed"
        ],
        "consult_doctor": "Consult a doctor if: symptoms don't improve with over-the-counter treatments, infection spreads, or you have a weakened immune system."
    },
    "DrugEruption": {
        "description": "Skin reactions caused by medications, ranging from mild rashes to severe allergic reactions.",
        "resources": [
            "American Academy of Dermatology: Drug Reactions",
            "Mayo Clinic: Drug Allergy",
            "DermNet NZ: Drug Eruptions"
        ],
        "treatment_recommendations": [
            "Stop taking the suspected medication (with doctor's approval)",
            "Use cool compresses to soothe skin",
            "Apply calamine lotion or hydrocortisone cream",
            "Take antihistamines for itching",
            "Avoid scratching to prevent infection"
        ],
        "consult_doctor": "Consult a doctor immediately if: rash is severe, covers large areas, involves blisters, or is accompanied by fever, difficulty breathing, or swelling."
    },
    "Eczema": {
        "description": "A chronic inflammatory skin condition causing dry, itchy, and inflamed skin.",
        "resources": [
            "National Eczema Association",
            "American Academy of Dermatology: Eczema",
            "Mayo Clinic: Atopic Dermatitis"
        ],
        "treatment_recommendations": [
            "Moisturize skin regularly with fragrance-free creams",
            "Use gentle, soap-free cleansers",
            "Avoid known triggers (allergens, irritants, stress)",
            "Take lukewarm baths and apply moisturizer immediately after",
            "Use topical corticosteroids or calcineurin inhibitors as prescribed"
        ],
        "consult_doctor": "Consult a dermatologist if: eczema is severe, affects daily life, doesn't respond to over-the-counter treatments, or shows signs of infection."
    },
    "Infestations_Bites": {
        "description": "Skin conditions caused by insect bites, mites, or other parasites.",
        "resources": [
            "CDC: Parasitic Diseases",
            "American Academy of Dermatology: Insect Bites",
            "Mayo Clinic: Scabies"
        ],
        "treatment_recommendations": [
            "Clean bite areas with soap and water",
            "Apply calamine lotion or hydrocortisone cream",
            "Use antihistamines for itching",
            "Avoid scratching to prevent infection",
            "For scabies or lice, follow specific treatment protocols with medicated creams"
        ],
        "consult_doctor": "Consult a doctor if: bites are severe, show signs of infection (redness, pus, fever), or if you suspect scabies, lice, or other parasitic infestations."
    },
    "Lichen": {
        "description": "A chronic inflammatory skin condition with characteristic flat-topped, purple, itchy bumps.",
        "resources": [
            "American Academy of Dermatology: Lichen Planus",
            "DermNet NZ: Lichen Planus",
            "Mayo Clinic: Lichen Planus"
        ],
        "treatment_recommendations": [
            "Avoid scratching affected areas",
            "Use gentle, fragrance-free skincare products",
            "Apply cool compresses for relief",
            "Topical corticosteroids or calcineurin inhibitors may be prescribed",
            "For severe cases, oral medications or light therapy may be recommended"
        ],
        "consult_doctor": "Consult a dermatologist for proper diagnosis and treatment. Lichen planus can be chronic and may require medical management."
    },
    "Lupus": {
        "description": "An autoimmune disease that can cause skin rashes, particularly a butterfly-shaped rash on the face.",
        "resources": [
            "Lupus Foundation of America",
            "American Academy of Dermatology: Lupus",
            "Mayo Clinic: Lupus"
        ],
        "treatment_recommendations": [
            "Protect skin from sun exposure (critical for lupus)",
            "Use broad-spectrum sunscreen SPF 50+ daily",
            "Wear protective clothing and hats",
            "Follow prescribed medications (antimalarials, corticosteroids)",
            "Manage stress and get adequate rest"
        ],
        "consult_doctor": "Consult a rheumatologist or dermatologist immediately. Lupus is a serious autoimmune condition requiring medical management and monitoring."
    },
    "Moles": {
        "description": "Common skin growths, usually brown or black, that can appear anywhere on the skin.",
        "resources": [
            "American Academy of Dermatology: Moles",
            "Skin Cancer Foundation: Moles",
            "Mayo Clinic: Moles"
        ],
        "treatment_recommendations": [
            "Monitor moles using the ABCDE rule (Asymmetry, Border, Color, Diameter, Evolving)",
            "Protect from sun exposure",
            "Regular self-examinations",
            "Most moles don't require treatment",
            "Surgical removal if suspicious or for cosmetic reasons"
        ],
        "consult_doctor": "Consult a dermatologist if: a mole changes in size, shape, or color; has irregular borders; is asymmetrical; or if you notice new moles after age 30."
    },
    "Psoriasis": {
        "description": "A chronic autoimmune condition causing rapid skin cell growth, resulting in thick, scaly patches.",
        "resources": [
            "National Psoriasis Foundation",
            "American Academy of Dermatology: Psoriasis",
            "Mayo Clinic: Psoriasis"
        ],
        "treatment_recommendations": [
            "Moisturize regularly to reduce scaling",
            "Use gentle, fragrance-free products",
            "Take lukewarm baths with added oils",
            "Topical treatments (corticosteroids, vitamin D analogs)",
            "For moderate to severe cases, systemic medications or biologics may be needed"
        ],
        "consult_doctor": "Consult a dermatologist if: psoriasis is widespread, affects quality of life, or doesn't respond to over-the-counter treatments. Regular monitoring recommended."
    },
    "Rosacea": {
        "description": "A chronic skin condition causing facial redness, visible blood vessels, and sometimes bumps.",
        "resources": [
            "National Rosacea Society",
            "American Academy of Dermatology: Rosacea",
            "Mayo Clinic: Rosacea"
        ],
        "treatment_recommendations": [
            "Identify and avoid triggers (spicy foods, alcohol, sun, stress)",
            "Use gentle, fragrance-free skincare products",
            "Protect from sun with SPF 30+ sunscreen",
            "Topical medications (metronidazole, azelaic acid)",
            "Oral antibiotics may be prescribed for moderate cases"
        ],
        "consult_doctor": "Consult a dermatologist if: rosacea is persistent, affects daily life, or doesn't improve with lifestyle changes. Professional treatment can help manage symptoms."
    },
    "Seborrh_Keratoses": {
        "description": "Benign, wart-like growths that appear as brown, black, or tan growths on the skin.",
        "resources": [
            "American Academy of Dermatology: Seborrheic Keratoses",
            "Mayo Clinic: Seborrheic Keratoses",
            "DermNet NZ: Seborrheic Keratosis"
        ],
        "treatment_recommendations": [
            "No treatment necessary unless they cause discomfort",
            "Monitor for changes (though rare)",
            "Surgical removal or cryotherapy if they become irritated or for cosmetic reasons",
            "Protect from sun exposure"
        ],
        "consult_doctor": "Consult a dermatologist if: growths change in appearance, become irritated, or you want them removed for cosmetic reasons."
    },
    "SkinCancer": {
        "description": "Abnormal growth of skin cells, most often caused by sun exposure. Early detection is crucial.",
        "resources": [
            "Skin Cancer Foundation",
            "American Academy of Dermatology: Skin Cancer",
            "American Cancer Society: Skin Cancer"
        ],
        "treatment_recommendations": [
            "Immediate medical evaluation required",
            "Protect from sun exposure with SPF 30+ sunscreen",
            "Regular skin examinations",
            "Treatment depends on type and stage (surgery, radiation, chemotherapy)",
            "Follow dermatologist's treatment plan strictly"
        ],
        "consult_doctor": "Consult a dermatologist IMMEDIATELY. Skin cancer requires prompt medical attention. Early detection and treatment are critical for successful outcomes."
    },
    "Sun_Sunlight_Damage": {
        "description": "Damage to the skin caused by long-term sun exposure, including wrinkles, age spots, and texture changes.",
        "resources": [
            "American Academy of Dermatology: Sun Damage",
            "Skin Cancer Foundation: Sun Protection",
            "Mayo Clinic: Photoaging"
        ],
        "treatment_recommendations": [
            "Use broad-spectrum sunscreen SPF 30+ daily",
            "Wear protective clothing and wide-brimmed hats",
            "Avoid peak sun hours (10 AM - 4 PM)",
            "Use retinoids or alpha-hydroxy acids as recommended",
            "Consider professional treatments (laser therapy, chemical peels)"
        ],
        "consult_doctor": "Consult a dermatologist for: evaluation of sun damage, treatment options, and regular skin cancer screenings."
    },
    "Tinea": {
        "description": "Fungal skin infections (ringworm) that can affect various parts of the body.",
        "resources": [
            "CDC: Fungal Diseases",
            "American Academy of Dermatology: Ringworm",
            "Mayo Clinic: Ringworm"
        ],
        "treatment_recommendations": [
            "Keep affected areas clean and dry",
            "Use antifungal creams (clotrimazole, terbinafine)",
            "Wash clothing and bedding regularly",
            "Avoid sharing personal items",
            "For severe or persistent cases, oral antifungal medications may be needed"
        ],
        "consult_doctor": "Consult a doctor if: infection doesn't improve with over-the-counter treatments, spreads to other areas, or affects nails or scalp."
    },
    "Unknown_Normal": {
        "description": "Normal, healthy skin with no apparent abnormalities.",
        "resources": [
            "American Academy of Dermatology: Healthy Skin",
            "Mayo Clinic: Skin Care Basics"
        ],
        "treatment_recommendations": [
            "Maintain good skincare routine",
            "Use sunscreen daily",
            "Stay hydrated",
            "Eat a balanced diet",
            "Get regular exercise"
        ],
        "consult_doctor": "Regular skin check-ups are recommended. Consult a dermatologist if you notice any changes in your skin."
    },
    "Vascular_Tumors": {
        "description": "Benign growths involving blood vessels, such as hemangiomas or angiomas.",
        "resources": [
            "American Academy of Dermatology: Vascular Birthmarks",
            "Mayo Clinic: Hemangioma",
            "DermNet NZ: Vascular Tumors"
        ],
        "treatment_recommendations": [
            "Most vascular tumors are benign and don't require treatment",
            "Monitor for changes in size or appearance",
            "Protect from trauma to prevent bleeding",
            "Laser therapy or surgical removal may be options if needed",
            "Follow dermatologist's recommendations"
        ],
        "consult_doctor": "Consult a dermatologist for: proper diagnosis, monitoring, and treatment options if the tumor causes problems or changes."
    },
    "Vasculitis": {
        "description": "Inflammation of blood vessels that can cause skin lesions, rashes, and other symptoms.",
        "resources": [
            "Vasculitis Foundation",
            "American College of Rheumatology: Vasculitis",
            "Mayo Clinic: Vasculitis"
        ],
        "treatment_recommendations": [
            "Follow prescribed medications (corticosteroids, immunosuppressants)",
            "Protect skin from trauma",
            "Manage underlying conditions",
            "Regular medical monitoring",
            "Report any new symptoms to your doctor"
        ],
        "consult_doctor": "Consult a rheumatologist or dermatologist immediately. Vasculitis is a serious condition requiring medical management and monitoring."
    },
    "Vitiligo": {
        "description": "A condition causing loss of skin color in patches due to destruction of melanocytes.",
        "resources": [
            "American Vitiligo Research Foundation",
            "American Academy of Dermatology: Vitiligo",
            "Mayo Clinic: Vitiligo"
        ],
        "treatment_recommendations": [
            "Protect depigmented areas from sun exposure",
            "Use sunscreen SPF 30+ on affected areas",
            "Topical corticosteroids or calcineurin inhibitors",
            "Light therapy (phototherapy) may be recommended",
            "Cosmetic cover-ups can help with appearance"
        ],
        "consult_doctor": "Consult a dermatologist for: proper diagnosis, treatment options, and management strategies. Vitiligo is a chronic condition that may require ongoing care."
    },
    "Warts": {
        "description": "Small, rough growths caused by the human papillomavirus (HPV).",
        "resources": [
            "American Academy of Dermatology: Warts",
            "Mayo Clinic: Warts",
            "WebMD: Warts Guide"
        ],
        "treatment_recommendations": [
            "Don't pick or scratch warts to prevent spreading",
            "Keep warts covered with bandages",
            "Use salicylic acid treatments",
            "Cryotherapy (freezing) by a doctor",
            "For persistent warts, other treatments like laser or surgical removal"
        ],
        "consult_doctor": "Consult a dermatologist if: warts are painful, spreading, or don't respond to over-the-counter treatments. Professional removal may be needed."
    }
}


def get_disease_info(disease_name: str) -> dict:
    """
    Get disease information including description, resources, treatment recommendations,
    and when to consult a doctor.
    
    Args:
        disease_name: Name of the disease (should match keys in DISEASE_INFO)
    
    Returns:
        Dictionary with disease information, or default if not found
    """
    # Try exact match first
    if disease_name in DISEASE_INFO:
        return DISEASE_INFO[disease_name]
    
    # Try case-insensitive match
    disease_name_lower = disease_name.lower()
    for key, value in DISEASE_INFO.items():
        if key.lower() == disease_name_lower:
            return value
    
    # Return default information if not found
    return {
        "description": "Information about this condition is being updated.",
        "resources": [
            "American Academy of Dermatology",
            "Mayo Clinic: Dermatology",
            "WebMD: Skin Conditions"
        ],
        "treatment_recommendations": [
            "Consult a dermatologist for proper diagnosis and treatment",
            "Protect skin from sun exposure",
            "Use gentle skincare products",
            "Monitor for any changes"
        ],
        "consult_doctor": "Consult a qualified dermatologist for proper evaluation and treatment recommendations."
    }

