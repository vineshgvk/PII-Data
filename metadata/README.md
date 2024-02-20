# METADATA DESCRIPTION:

The following seven types of PII are to be identified:

**NAME_STUDENT:** Refers to the full or partial name of a student mentioned in the essay. This excludes names of instructors, authors, or other individuals not associated with student identities.

**EMAIL:** Represents a student's email address mentioned in the essay.

**USERNAME:** Denotes a student's username on any platform, if mentioned in the essay.

**ID_NUM:** Indicates any numerical or character sequence that could potentially identify a student, such as a student ID or a social security number.

**PHONE_NUM:** Represents a phone number associated with a student, if provided in the essay.

**URL_PERSONAL:** Refers to a URL that might be used to identify a student, if included in the essay.

**STREET_ADDRESS:** Denotes a full or partial street address associated with the student, such as their home address, if mentioned in the essay.


**Dataset description:**

* `(int)`: the index of the essay
    
* `document` (int): an integer ID of the essay

* `full_text` (string): a UTF-8 representation of the essay

* `tokens` (list)

    * (string): a string representation of each token
        
* `trailing_whitespace` (list)

    * (bool): a boolean value indicating whether each token is followed by whitespace.

* `labels` (list) [training data only]

    * (string): a token label in BIO format