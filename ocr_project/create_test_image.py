# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw, ImageFont

# Create a larger, higher quality image
img = Image.new('RGB', (1200, 1600), 'white')
draw = ImageDraw.Draw(img)

# Try to use a better font, fall back to default if not available
try:
    font_title = ImageFont.truetype("arial.ttf", 28)
    font_text = ImageFont.truetype("arial.ttf", 20)
except:
    font_title = ImageFont.load_default()
    font_text = ImageFont.load_default()

# Document content
y = 50
lines = [
    ("PERSONAL DETAILS FORM", font_title, 50),
    ("=" * 60, font_text, 30),
    ("", font_text, 20),
    ("APPLICANT INFORMATION", font_text, 40),
    ("Name: Rajesh Kumar Sharma", font_text, 35),
    ("Father's Name: Vijay Kumar Sharma", font_text, 35),
    ("Date of Birth: 15/08/1990", font_text, 35),
    ("Gender: Male", font_text, 35),
    ("", font_text, 20),
    ("RESIDENTIAL ADDRESS", font_text, 40),
    ("Address: House No 234, Green Park Extension, Sector 12, New Delhi", font_text, 35),
    ("Pincode: 110016", font_text, 35),
    ("State: Delhi", font_text, 35),
    ("", font_text, 20),
    ("CONTACT INFORMATION", font_text, 40),
    ("Mobile: 9876543210", font_text, 35),
    ("Email: rajesh.sharma@email.com", font_text, 35),
    ("Phone: 8765432109", font_text, 35),
    ("", font_text, 20),
    ("IDENTIFICATION NUMBERS", font_text, 40),
    ("Aadhaar: 1234 5678 9012", font_text, 35),
    ("PAN: ABCDE1234F", font_text, 35),
    ("Bank Account: 12345678901", font_text, 35),
    ("Passport: M1234567", font_text, 35),
    ("", font_text, 20),
    ("OTHER DETAILS (Testing Context Detection)", font_text, 40),
    ("Reference Number: 9988776655", font_text, 35),
    ("Transaction ID: 110025", font_text, 35),
    ("Order Number: 556677", font_text, 35),
    ("", font_text, 20),
    ("PASSWORD INFORMATION", font_text, 40),
    ("Password: MySecret123!", font_text, 35),
    ("PIN: 1234", font_text, 35),
]

current_y = y
for text, font, spacing in lines:
    draw.text((80, current_y), text, fill='black', font=font)
    current_y += spacing

# Save the image
img.save('test_document_quality.png')
print("High quality test document created: test_document_quality.png")
print(f"Image size: {img.size[0]}x{img.size[1]} pixels")
