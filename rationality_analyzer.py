import re
from typing import Dict, Tuple, List, Any

class RationalityAnalyzer:
    def __init__(self):
        """
        Инициализация анализатора рациональности.
        Все значения теперь целые числа (количество слов, а не веса).
        """
        # Эмоциональные маркеры (только слова, без весов)
        self.emotional_markers = {
            'positive': [
                'joy', 'happiness', 'love', 'excitement', 'passion',
                'hope', 'pride', 'satisfaction', 'delight', 'pleasure',
                'amazing', 'incredible', 'fantastic', 'wonderful', 'awesome',
                'brilliant', 'excellent', 'perfect', 'outstanding', 'marvelous',
                'thrilled', 'ecstatic', 'blissful', 'elated', 'enthusiastic',
                'optimistic', 'grateful', 'thankful', 'inspired', 'motivated',
                'determined', 'confident'
            ],
            'negative': [
                'anger', 'fear', 'sadness', 'anxiety', 'disappointment',
                'hate', 'irritation', 'resentment', 'annoyance', 'frustration',
                'worry', 'doubt', 'uncertainty', 'terrible', 'horrible',
                'awful', 'disgusting', 'horrifying', 'devastating', 'catastrophic',
                'depressed', 'miserable', 'unhappy', 'disgusted', 'terrified',
                'scared', 'panicked', 'desperate', 'helpless', 'hopeless',
                'ashamed', 'guilty'
            ],
            'subjective': [
                'I think', 'I feel', 'in my opinion', 'it seems to me',
                'I believe', 'I hope', 'I suppose', 'I guess', 'I assume',
                'from my perspective', 'I am certain', 'I am sure', 'I am convinced',
                'I know', 'I understand', 'I realize', 'in my view',
                'as far as I can tell', 'to my knowledge', 'personally',
                'for me', 'in my experience'
            ],
            'intensifiers': [
                'absolutely', 'completely', 'totally', 'entirely', 'fully',
                'utterly', 'extremely', 'incredibly', 'amazingly', 'perfectly',
                'definitely', 'certainly', 'surely', 'obviously', 'clearly',
                'undeniably', 'wholly', 'thoroughly', 'profoundly', 'intensely',
                'immensely'
            ],
            'emotional_verbs': [
                'feel', 'love', 'hate', 'enjoy', 'desire',
                'want', 'need', 'crave', 'fear', 'worry',
                'doubt', 'hope', 'believe', 'trust', 'suspect',
                'adore', 'cherish', 'treasure', 'loathe', 'despise',
                'detest', 'yearn', 'long', 'appreciate', 'value',
                'respect'
            ]
        }
        
        # Логические связки
        self.logical_connectors = [
            'therefore', 'thus', 'consequently', 'as a result', 'if...then',
            'since', 'because', 'due to', 'as a consequence', 'in conclusion',
            'hence', 'accordingly', 'so', 'ergo', 'it follows that',
            'this implies', 'leads to', 'indicates that', 'suggests that',
            'demonstrates that', 'furthermore', 'moreover', 'additionally',
            'in addition', 'also', 'besides', 'similarly', 'likewise',
            'in the same way', 'on the other hand', 'however', 'nevertheless',
            'nonetheless', 'conversely', 'in contrast', 'whereas', 'although',
            'even though', 'despite', 'in spite of', 'while'
        ]
        
        # Научные термины
        self.scientific_terms = [
            'analysis', 'research', 'data', 'result', 'method',
            'hypothesis', 'theory', 'experiment', 'observation', 'conclusion',
            'evidence', 'study', 'findings', 'measurement', 'variable',
            'parameter', 'correlation', 'causation', 'statistical', 'significant',
            'valid', 'reliable', 'accurate', 'precise', 'systematic',
            'empirical', 'quantitative', 'qualitative', 'methodology', 'procedure',
            'protocol', 'framework', 'model', 'algorithm', 'simulation',
            'validation', 'verification', 'replication', 'peer-reviewed',
            'controlled', 'experimental', 'observational', 'longitudinal',
            'cross-sectional', 'randomized', 'double-blind', 'placebo',
            'control group', 'sample', 'population', 'cohort', 'bias',
            'confounding', 'statistical significance', 'p-value',
            'confidence interval', 'standard deviation', 'mean', 'median',
            'mode', 'variance', 'correlation coefficient', 'regression',
            'meta-analysis', 'systematic review', 'literature review'
        ]
        
        # Компилируем паттерны для эффективности
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Компилирует регулярные выражения для быстрого поиска"""
        self.patterns = {}
        
        # Компилируем паттерны эмоциональных маркеров
        for category, markers in self.emotional_markers.items():
            self.patterns[f'{category}_patterns'] = [
                re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
                for word in markers
            ]
        
        # Компилируем паттерны логических связок
        self.patterns['logical_patterns'] = [
            re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
            for word in self.logical_connectors
        ]
        
        # Компилируем паттерны научных терминов
        self.patterns['scientific_patterns'] = [
            re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
            for word in self.scientific_terms
        ]
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Основной метод анализа текста.
        
        Аргументы:
            text (str): Текст для анализа
            
        Возвращает:
            Dict: Словарь с результатами анализа (все значения целые числа)
        """
        if not text or not isinstance(text, str):
            raise ValueError("Текст должен быть непустой строкой")
        
        text_lower = text.lower()
        
        # 1. Считаем эмоциональные маркеры (целые числа)
        emotional_count, emotional_details = self._count_emotional_markers(text_lower)
        
        # 2. Считаем логические связки (целые числа)
        logical_count, logical_details = self._count_logical_connectors(text_lower)
        
        # 3. Считаем научные термины (целые числа)
        scientific_count, scientific_details = self._count_scientific_terms(text_lower)
        
        # 4. Анализируем структуру предложений
        structure_score, structure_details = self._analyze_sentence_structure(text)
        
        # 5. Вычисляем коэффициент рациональности
        rationality_score = self._calculate_rationality(
            emotional_count, logical_count, scientific_count, structure_score
        )
        
        # 6. Вычисляем детализированный разбор
        breakdown = self._calculate_breakdown(
            emotional_count, logical_count, scientific_count, structure_score
        )
        
        return {
            'rationality_score': rationality_score,
            'emotional_markers': {
                'count': emotional_count,
                'details': emotional_details
            },
            'logical_connectors': {
                'count': logical_count,
                'details': logical_details
            },
            'scientific_terms': {
                'count': scientific_count,
                'details': scientific_details
            },
            'structure_quality': {
                'score': structure_score,
                'details': structure_details
            },
            'analysis': self._generate_analysis(rationality_score),
            'breakdown': breakdown,
            'raw_scores': {
                'emotional': emotional_count,
                'logical': logical_count,
                'scientific': scientific_count,
                'structural': structure_score
            }
        }
    
    def _count_emotional_markers(self, text: str) -> Tuple[int, Dict]:
        """
        Считает эмоциональные маркеры в тексте (целые числа).
        
        Аргументы:
            text (str): Текст для анализа (в нижнем регистре)
            
        Возвращает:
            Tuple[int, Dict]: (общее количество, детали по категориям)
        """
        total_count = 0
        details = {
            'positive': 0,
            'negative': 0,
            'subjective': 0,
            'intensifiers': 0,
            'emotional_verbs': 0,
            'markers_found': []
        }
        
        # Считаем маркеры в каждой категории
        for category in ['positive', 'negative', 'subjective', 'intensifiers', 'emotional_verbs']:
            category_count = 0
            patterns = self.patterns.get(f'{category}_patterns', [])
            
            for pattern in patterns:
                try:
                    matches = pattern.findall(text)
                    count = len(matches)
                    
                    if count > 0:
                        category_count += count
                        total_count += count
                        details['markers_found'].extend([
                            {'word': match, 'category': category}
                            for match in matches
                        ])
                except Exception as e:
                    continue
            
            details[category] = category_count
        
        return total_count, details
    
    def _count_logical_connectors(self, text: str) -> Tuple[int, Dict]:
        """
        Считает логические связки в тексте (целые числа).
        
        Аргументы:
            text (str): Текст для анализа (в нижнем регистре)
            
        Возвращает:
            Tuple[int, Dict]: (общее количество, детали)
        """
        total_count = 0
        details = {'connectors_found': []}
        
        # Проверяем реальные логические связки
        for pattern in self.patterns['logical_patterns']:
            try:
                matches = pattern.findall(text)
                count = len(matches)
                
                if count > 0:
                    total_count += count
                    details['connectors_found'].extend([
                        {'connector': match}
                        for match in matches
                    ])
            except Exception as e:
                continue
        
        return total_count, details
    
    def _count_scientific_terms(self, text: str) -> Tuple[int, Dict]:
        """
        Считает научные термины в тексте (целые числа).
        
        Аргументы:
            text (str): Текст для анализа (в нижнем регистре)
            
        Возвращает:
            Tuple[int, Dict]: (общее количество, детали)
        """
        total_count = 0
        details = {'terms_found': []}
        
        for pattern in self.patterns['scientific_patterns']:
            try:
                matches = pattern.findall(text)
                count = len(matches)
                
                if count > 0:
                    total_count += count
                    details['terms_found'].extend([
                        {'term': match}
                        for match in matches
                    ])
            except Exception as e:
                continue
        
        return total_count, details
    
    def _analyze_sentence_structure(self, text: str) -> Tuple[float, Dict]:
        """
        Анализирует качество структуры предложений.
        
        Аргументы:
            text (str): Исходный текст
            
        Возвращает:
            Tuple[float, Dict]: (оценка структуры, детали)
        """
        # Разделяем на предложения
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0, {'sentence_count': 0, 'avg_length': 0}
        
        # Вычисляем среднюю длину предложения
        word_counts = [len(s.split()) for s in sentences]
        avg_length = sum(word_counts) / len(sentences)
        
        # Вычисляем сложность
        long_sentences = sum(1 for wc in word_counts if wc > 25)
        short_sentences = sum(1 for wc in word_counts if wc < 5)
        
        # Оценка: 10-25 слов = оптимально для рациональности
        if 10 <= avg_length <= 25:
            base_score = 1.0
        elif 5 <= avg_length < 10 or 25 < avg_length <= 40:
            base_score = 0.7
        elif 3 <= avg_length < 5 or 40 < avg_length <= 60:
            base_score = 0.4
        else:
            base_score = 0.1
        
        # Штраф за слишком много коротких или длинных предложений
        penalty = 0.0
        if long_sentences > len(sentences) * 0.3:
            penalty += 0.2
        if short_sentences > len(sentences) * 0.3:
            penalty += 0.2
        
        final_score = max(0.0, base_score - penalty)
        
        details = {
            'sentence_count': len(sentences),
            'avg_length': round(avg_length, 1),
            'long_sentences': long_sentences,
            'short_sentences': short_sentences,
            'penalty': round(penalty, 2)
        }
        
        return round(final_score, 2), details
    
    def _calculate_rationality(self, emo: int, log: int, sci: int, struct: float) -> float:
        """
        Вычисляет коэффициент рациональности.
        
        Аргументы:
            emo (int): Количество эмоциональных маркеров
            log (int): Количество логических связок
            sci (int): Количество научных терминов
            struct (float): Оценка структуры
            
        Возвращает:
            float: Оценка рациональности (0-100%)
        """
        # Базовый балл: меньше эмоций = выше рациональность
        # Штраф за эмоции: 8 баллов за каждый эмоциональный маркер
        base_score = max(0.0, 100.0 - (emo * 8.0))
        
        # Бонусы за логику и науку
        logic_bonus = log * 2.0  # 2 балла за каждую логическую связку
        science_bonus = sci * 1.5  # 1.5 балла за каждый научный термин
        structure_bonus = struct * 10.0  # Максимум 10 баллов за структуру
        
        # Ограничения на бонусы (чтобы предотвратить перекомпенсацию)
        logic_bonus = min(logic_bonus, 15.0)
        science_bonus = min(science_bonus, 20.0)
        structure_bonus = min(structure_bonus, 10.0)
        
        # Общий бонус
        total_bonus = logic_bonus + science_bonus + structure_bonus
        
        # Финальный балл (ограничен 0-100)
        final_score = max(0.0, min(100.0, base_score + total_bonus))
        
        return round(final_score, 2)
    
    def _calculate_breakdown(self, emo: int, log: int, sci: int, struct: float) -> Dict:
        """
        Вычисляет детализированный разбор компонентов оценки.
        
        Аргументы:
            emo (int): Количество эмоциональных маркеров
            log (int): Количество логических связок
            sci (int): Количество научных терминов
            struct (float): Оценка структуры
            
        Возвращает:
            Dict: Разбор компонентов
        """
        base_score = max(0.0, 100.0 - (emo * 8.0))
        logic_bonus = min(log * 2.0, 15.0)
        science_bonus = min(sci * 1.5, 20.0)
        structure_bonus = min(struct * 10.0, 10.0)
        
        return {
            'base_score': round(base_score, 2),
            'emotional_penalty': round(emo * 8.0, 2),
            'logical_bonus': round(logic_bonus, 2),
            'scientific_bonus': round(science_bonus, 2),
            'structure_bonus': round(structure_bonus, 2),
            'total_bonus': round(logic_bonus + science_bonus + structure_bonus, 2)
        }
    
    def _generate_analysis(self, score: float) -> str:
        """
        Генерирует текстовый анализ на основе оценки.
        
        Аргументы:
            score (float): Оценка рациональности
            
        Возвращает:
            str: Текстовый анализ
        """
        if score >= 95:
            return "Text is exceptionally rational. Virtually no emotional content. Excellent logical structure and scientific rigor."
        elif score >= 90:
            return "Text is highly rational. Emotions are virtually absent. Excellent logical structure."
        elif score >= 85:
            return "Text is very rational. Minimal emotional influence. Strong logical foundation."
        elif score >= 80:
            return "Text is predominantly rational. Very minor emotional elements. Good logical flow."
        elif score >= 75:
            return "Text is mostly rational. Some emotional content detected but well-controlled."
        elif score >= 70:
            return "Text is moderately rational. Noticeable emotional influence but still logical."
        elif score >= 65:
            return "Text has moderate rationality. Emotional content is present and affects logic."
        elif score >= 60:
            return "Text contains significant emotional content. Rationality is compromised."
        elif score >= 55:
            return "Text is emotionally charged. Rationality is low with weak logical connections."
        elif score >= 50:
            return "Text is highly emotional. Rationality is severely limited."
        elif score >= 45:
            return "Text is dominated by emotions. Rationality is almost absent."
        elif score >= 40:
            return "Text is overwhelmingly emotional. No clear logical structure."
        elif score >= 30:
            return "Text is entirely emotional with minimal rational elements."
        elif score >= 20:
            return "Text is purely emotional expression with no rational content."
        else:
            return "Text is completely emotional. Zero rationality detected."

# Тестирование
if __name__ == "__main__":
    analyzer = RationalityAnalyzer()
    
    # Пример 1: Высокая рациональность
    text1 = """
    Data analysis confirms the hypothesis. 
    The experiment was conducted over 30 days with 500 participants. 
    Results demonstrate a statistically significant correlation between variables. 
    Therefore, the conclusion is reliable and valid.
    """
    
    # Пример 2: Эмоциональный текст
    text2 = """
    This is absolutely amazing! I feel like this discovery will change everything! 
    I'm so excited that I finally found the right path. 
    Nobody can doubt this! This is the ultimate truth, I'm certain!
    """
    
    # Пример 3: Смешанный текст
    text3 = """
    I think this method works well. 
    The data shows positive results and strong evidence. 
    However, I'm a bit worried about potential limitations.
    """
    
    print("=" * 60)
    print("HIGHLY RATIONAL TEXT:")
    print("=" * 60)
    result1 = analyzer.analyze(text1)
    print(f"Rationality Score: {result1['rationality_score']}%")
    print(f"Analysis: {result1['analysis']}")
    print(f"Emotional markers: {result1['emotional_markers']['count']}")
    print(f"Logical connectors: {result1['logical_connectors']['count']}")
    print(f"Scientific terms: {result1['scientific_terms']['count']}")
    print(f"Breakdown: {result1['breakdown']}")
    
    print("\n" + "=" * 60)
    print("EMOTIONAL TEXT:")
    print("=" * 60)
    result2 = analyzer.analyze(text2)
    print(f"Rationality Score: {result2['rationality_score']}%")
    print(f"Analysis: {result2['analysis']}")
    print(f"Emotional markers: {result2['emotional_markers']['count']}")
    print(f"Logical connectors: {result2['logical_connectors']['count']}")
    print(f"Scientific terms: {result2['scientific_terms']['count']}")
    print(f"Breakdown: {result2['breakdown']}")
    
    print("\n" + "=" * 60)
    print("MIXED TEXT:")
    print("=" * 60)
    result3 = analyzer.analyze(text3)
    print(f"Rationality Score: {result3['rationality_score']}%")
    print(f"Analysis: {result3['analysis']}")
    print(f"Emotional markers: {result3['emotional_markers']['count']}")
    print(f"Logical connectors: {result3['logical_connectors']['count']}")
    print(f"Scientific terms: {result3['scientific_terms']['count']}")
    print(f"Breakdown: {result3['breakdown']}")