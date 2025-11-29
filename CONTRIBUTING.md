# Contributing to Twitter NER System

Thank you for your interest in contributing to the Twitter Named Entity Recognition System! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Screenshots (if applicable)
- Environment details (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:
- Clear description of the enhancement
- Use case and benefits
- Possible implementation approach
- Any relevant examples

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the code style guidelines
   - Add tests if applicable
   - Update documentation

4. **Commit your changes**
   ```bash
   git commit -m "feat: add amazing feature"
   ```
   
   Use conventional commits:
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation changes
   - `style:` formatting changes
   - `refactor:` code refactoring
   - `test:` adding tests
   - `chore:` maintenance tasks

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Provide a clear description
   - Reference any related issues
   - Wait for review

## 📝 Code Style Guidelines

### Python
- Follow PEP 8 style guide
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use meaningful variable names
- Add docstrings to all functions/classes

### Example
```python
def predict_entities(text: str) -> List[Dict[str, str]]:
    """
    Predict named entities in the given text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of dictionaries containing word, entity, and color
    """
    # Implementation
    pass
```

### Documentation
- Update README.md for new features
- Add inline comments for complex logic
- Keep documentation up-to-date

## 🧪 Testing

Before submitting a PR:
1. Test your changes locally
2. Ensure backend starts without errors
3. Verify frontend functionality
4. Check for console errors
5. Test edge cases

## 📋 Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/twitter-ner-system.git
cd twitter-ner-system

# Install dependencies
pip install -r requirements.txt

# Start backend
cd project/backend
python -m uvicorn main:app --port 8000 --reload

# Start frontend (new terminal)
cd project/frontend
streamlit run app.py --server.port 8501
```

## 🎯 Areas for Contribution

### High Priority
- [ ] Add unit tests
- [ ] Improve model training speed
- [ ] Add more entity types
- [ ] Implement model comparison feature
- [ ] Add export functionality (CSV, JSON)

### Medium Priority
- [ ] Dark mode toggle
- [ ] Multi-language support
- [ ] Batch processing for multiple texts
- [ ] Model performance metrics
- [ ] API rate limiting

### Low Priority
- [ ] Additional visualization options
- [ ] Custom color schemes
- [ ] Keyboard shortcuts
- [ ] Mobile responsiveness improvements

## 📞 Questions?

If you have questions, feel free to:
- Open an issue
- Contact: ratneshsingh181998@gmail.com
- LinkedIn: [Ratnesh Singh](https://www.linkedin.com/in/ratnesh-singh-a01a47193/)

## 🙏 Thank You!

Your contributions make this project better for everyone!

---

**RATNESH SINGH**  
Project Maintainer
