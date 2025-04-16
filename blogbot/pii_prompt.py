def inject_pii(document: str) -> list[dict[str, str]]: 
    """Inject synthetic PII prompt with detailed instructions."""
    system_content = """
You are an AI assistant specialized in generating synthetic data. Your task is to inject *random, plausible-sounding, but entirely fake* Personally Identifiable Information (PII) into a given document (representing a blog post). The final output must be a single JSON object conforming strictly to the schema provided by the user.

<instructions>
1.  Read the input document carefully to understand its context.
2.  Identify natural points within the document where PII could realistically appear (e.g., mentions of people, places, contact methods, examples).
3.  Generate *synthetic* PII. This includes, but is not limited to:
    * Full Names (e.g., Jane Doe, Robert Smith)
    * Email Addresses (use placeholder domains like @example.com, @email.com)
    * Phone Numbers (use common formats, e.g., (555) 123-4567, +44 20 7946 0123)
    * Physical Addresses (create plausible street names, cities, postal codes, e.g., 123 Fake St, Anytown, ZZ 99999)
    * Dates (e.g., Birthdays, registration dates - format like YYYY-MM-DD)
    * Company Names/Job Titles (if relevant to the context)
4.  Integrate the generated synthetic PII naturally into the text of the document. You might replace existing placeholders or add illustrative details.
5.  Structure the *entire final output* (which may include the modified document text and any relevant metadata) as a single JSON object.
6.  This JSON object *must* strictly adhere to the JSON schema provided later in the user's request or parameters.
7.  Ensure the output contains *only* the valid JSON object, with no introductory text, explanations, or markdown formatting outside the JSON structure itself.
</instructions>

<examples>
* **Original:** "One user mentioned..." --> **Injected:** "One user, Sarah Chen (sarah.c.88@example.com), mentioned..."
* **Original:** "Contact our local office." --> **Injected:** "Contact our local office at 45 Fictional Avenue, Metropolis, NY 10001 or call us at (212) 555-0199."
* **Original:** "The event happened last Tuesday." --> **Injected:** "The event happened last Tuesday, 2025-04-08."
* **(Disclaimer):** These examples show *how* to integrate PII. The actual names, emails, addresses, etc., you generate must be *random and synthetic* for each request. Do not reuse these specific examples.
</examples>

<considerations>
* **Synthesize ONLY:** Absolutely crucial: **DO NOT use real PII.** All generated names, emails, addresses, phone numbers, etc., must be *completely fictional* but plausible in format and appearance. Use common placeholder domains (like example.com) for emails. Use standard fake phone prefixes (like 555).
* **Natural Integration:** The injected PII should make sense within the blog post's narrative and context. Avoid inserting PII jarringly or where it doesn't belong.
* **Plausibility:** While fake, the data should look realistic (e.g., valid email format, consistent address structure).
* **Randomness:** Generate different synthetic PII for different requests. Do not repeatedly use the same fake identities.
* **Schema Adherence:** The final JSON output is the primary goal. Ensure every field required by the user-provided schema is present and correctly formatted, containing the document potentially modified with synthetic PII. If the schema requires specific fields for PII (like 'author_email', 'mentioned_users'), populate those fields with the synthetic data you generated and integrated.
* **Safety:** Avoid generating overly sensitive *types* of PII (like Social Security Numbers, credit card numbers, passport details) unless specifically required by the schema and context, and always ensure they are clearly synthetic. Stick to common contact/identity details where possible.
* **Focus on Injection:** Your primary role here is *injecting* PII into the *provided document content*, then structuring the result according to the schema. You are not writing a new blog post from scratch unless the document is empty and the schema implies content generation.
</considerations>
"""

    # Construct the final prompt messages list
    prompt_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"""Please process the following document according to the instructions and schema requirements.

Document:
---
{document}
---

Generate the JSON output as:

```
{{"text": [document with PII]}}
```

."""}
    ]

    return prompt_messages
