from bs4 import BeautifulSoup

markup = '<a href="http://example.com/">\nI linked to <i>example.com</i>\n</a>'
soup = BeautifulSoup(markup, "html5lib")

print(soup.get_text())
# u'\nI linked to example.com\n'
print(soup.i.get_text())
# u'example.com'
print(soup.get_text("|"))
print(soup.get_text("|", strip=True))