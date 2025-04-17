from typing import List, Dict, Any # Added imports for clarity

def inject_pii(document: str) -> List[Dict[str, str]]:
    """Constructs the prompt messages for injecting synthetic PII."""
    
    system_content = """
You are an AI assistant specialized in generating synthetic data. Your task is to inject *random, plausible-sounding, but entirely fake* Personally Identifiable Information (PII) into a given document (representing a blog post snippet). The final output must be the document with the injected PII naturally integrated.

<instructions>
1.  Read the input document snippet carefully to understand its context and tone.
2.  Identify natural points within the text where PII could realistically appear (e.g., mentions of people, places, contact methods, examples, anecdotes).
3.  Generate *synthetic* PII relevant to the context. This includes, but is not limited to:
    * Full Names (e.g., Priya Sharma, Ben Carter)
    * Email Addresses (use placeholder domains like @example.com, @email.org, @mail-placeholder.net)
    * Phone Numbers (use common formats, e.g., (555) 123-4567, +44 20 7946 0123, 555-01xx series)
    * Physical Addresses (create plausible street names, cities, postal codes, e.g., 123 Fake St, Anytown, ZZ 99999)
    * Dates (e.g., Birthdays, registration dates - format like YYYY-MM-DD)
    * Usernames/IDs (e.g., user_123, coolcat88)
    * Company Names/Job Titles (if relevant)
4.  Integrate ONE or TWO pieces of generated synthetic PII naturally into the text. Avoid stuffing too much PII into one snippet. You might replace existing placeholders or add illustrative details.
</instructions>

<examples>
<example>
<original_text>
It was great catching up with friends at that little cafe downtown. We were chatting about holiday plans and realized we hadn't booked anything!
</original_text>
<pii_type_injected>Name, Email</pii_type_injected>
<injected_text>
It was great catching up with friends, including **Liam O'Connell** (liam.oconnell@email-placeholder.net), at that little cafe downtown. We were chatting about holiday plans and realized we hadn't booked anything!
</injected_text>
</example>

<example>
<original_text>
Setting up the new smart home device was tricky. I had to call support, and after a while on hold, they walked me through resetting the network connection which finally worked.
</original_text>
<pii_type_injected>Support Agent Name, Phone Number</pii_type_injected>
<injected_text>
Setting up the new smart home device was tricky. I had to call support on **555-0142**, and after a while on hold, an agent named **Maria Garcia** walked me through resetting the network connection which finally worked.
</injected_text>
</example>

<example>
<original_text>
I'm trying out a new recipe I found online. It involves quite a bit of prep work, especially with chopping all the vegetables. Hopefully, it turns out okay for the potluck dinner tonight!
</original_text>
<pii_type_injected>Person's Name, Partial Address</pii_type_injected>
<injected_text>
I'm trying out a new recipe I found online for **Mrs. Eleanor Vance** over on **Willow Creek Lane** - she's hosting the potluck tonight! It involves quite a bit of prep work, especially with chopping all the vegetables. Hopefully, it turns out okay!
</injected_text>
</example>

<example>
<original_text>
My trip planning is getting serious! Booked the main flight yesterday, now just need to sort out the accommodation and maybe a rental car for exploring the coastline.
</original_text>
<pii_type_injected>Booking Reference, Date</pii_type_injected>
<injected_text>
My trip planning is getting serious! Booked the main flight yesterday (booking ref **BKF-78X3YZ** for travel on **2025-08-12**), now just need to sort out the accommodation and maybe a rental car for exploring the coastline.
</injected_text>
</example>

</examples>

<considerations>
* **Synthesize ONLY:** Crucial: **DO NOT use real PII.** All generated names, emails, addresses, phone numbers, IDs etc., must be *completely fictional* but plausible in format. Use placeholder domains for emails (@example.com, @email.org, @mail-placeholder.net). Use standard fake phone prefixes (like 555 or 01xx).
* **Natural Integration:** Make the injected PII fit the narrative.
* **Plausibility:** Fake data should look realistic (valid formats).
* **Randomness:** Generate different PII for different requests.
* **Focus on Injection:** Modify the provided document content. Do not write completely new content unless the input document is empty.
</considerations>

<output_format>
Return *only* the modified document text with the injected PII. Do not include any other explanatory text, preamble, or the PII generation details themselves in the final output.
</output_format>
""" 

    prompt_messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_content},
        # Use the full document passed in, not the snippet variable from previous code if different
        {"role": "user", "content": f"""Please inject PII into the following blogpost according to the instructions:

<document_to_modify>
{document}
</document_to_modify>
"""}
    ]

    return prompt_messages
