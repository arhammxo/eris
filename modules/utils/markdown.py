import markdown
from markupsafe import Markup

def render_markdown(text):
    """
    Simple markdown rendering without external dependencies.
    Falls back to basic formatting if markdown package is unavailable.
    """
    if not text:
        return ""
    
    try:
        # Try using the markdown package if available
        import markdown
        from markupsafe import Markup
        
        html = markdown.markdown(
            text,
            extensions=[
                'markdown.extensions.tables',
                'markdown.extensions.fenced_code',
                'markdown.extensions.nl2br'
            ]
        )
        return Markup(html)
    except ImportError:
        # Fall back to simple regex-based formatting
        import re
        from markupsafe import Markup
        
        # Basic markdown formatting
        text = re.sub(r'^#\s+(.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
        text = re.sub(r'^##\s+(.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^###\s+(.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
        
        # Convert paragraphs (simple)
        text = '<p>' + text.replace('\n\n', '</p><p>') + '</p>'
        text = text.replace('<p></p>', '')
        
        return Markup(text)
    except Exception as e:
        # If anything goes wrong, return the plain text
        from markupsafe import escape
        return escape(text)