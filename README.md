# advanced-logic-detector
Advanced Logic Contradiction Detector - Анализатор рациональности и логических противоречий текста
# Advanced Logic Contradiction Detector

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![NLP](https://img.shields.io/badge/NLP-Transformers-blue.svg)](https://huggingface.co/docs/transformers/)

**А# 📝 Пример содержания README.md для вашего проекта

```markdown
# Advanced Logic Contradiction Detector

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/krisstern9-create/advanced-logic-detector)](https://github.com/krisstern9-create/advanced-logic-detector/issues)
[![GitHub stars](https://img.shields.io/github/stars/krisstern9-create/advanced-logic-detector)](https://github.com/krisstern9-create/advanced-logic-detector/stargazers)

## 📖 Описание

**Advanced Logic Contradiction Detector** - это инструмент для анализа текста на рациональность и обнаружения логических противоречий. 

Проект анализирует текст и предоставляет:
- Оценку рациональности (0-100%)
- Количество эмоциональных маркеров
- Количество логических связок
- Количество научных терминов
- Обнаруженные противоречия
- Рекомендации по улучшению

**Статус проекта:**  
✅ Основные функции работают  
⚠️ Требует доработки (см. раздел "Что требует доработки")

---

## ✅ Что работает идеально

- [x] **Анализ рациональности текста** - точный подсчёт эмоциональных маркеров
- [x] **Обнаружение логических противоречий** - через паттерны и NLI-модель
- [x] **Веб-интерфейс** - работает стабильно для коротких текстов
- [x] **Генерация отчетов** - детальная аналитика с рекомендациями

---

## ⚠️ Что требует доработки

- [ ] **Точность определения противоречий** - некоторые противоречия не обнаруживаются
- [ ] **Расчет рациональности** - иногда дает неточные результаты для сложных текстов
- [ ] **Обработка длинных текстов** - может замедляться при анализе текстов > 5000 символов
- [ ] **Оптимизация производительности** - первоначальная загрузка модели занимает 1-2 минуты

---

## 🛠 Установка и запуск

### Требования
- Python 3.8+
- 2 ГБ свободной памяти (для загрузки модели)
- 500 МБ свободного места на диске

### Установка зависимостей
```bash
pip install -r requirements.txt
