import re
import torch
import numpy as np
from transformers import pipeline
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import warnings

# Подавляем предупреждения transformers
warnings.filterwarnings('ignore')

class AdvancedContradictionDetector:
    def __init__(self):
        """
        Инициализация детектора противоречий.
        Загружает модель NLI и компилирует все паттерны для анализа.
        """
        # Загружаем предобученную модель NLI для семантического анализа противоречий
        print("🔄 Loading NLI model (MoritzLaurer/deberta-v3-base-mnli-fever-anli)...")
        try:
            self.nli_pipeline = pipeline(
                "text-classification",
                model="MoritzLaurer/deberta-v3-base-mnli-fever-anli",
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else None
            )
            print("✅ NLI model loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load NLI model: {e}")
            print("⚠️ Semantic analysis will be disabled")
            self.nli_pipeline = None
        
        # Компилируем паттерны
        self._compile_patterns()
        
        # Инициализируем кэш для ускорения
        self._init_cache()
    
    def _compile_patterns(self):
        """Компилирует регулярные выражения для быстрого поиска противоречий"""
        self.patterns = {
            'direct_contradictions': [
                # Всегда/никогда противоречия
                (re.compile(r'\b(always|constantly|permanently|without exception|invariably|absolutely|certainly|definitely|unfailingly|infallibly)\b', re.IGNORECASE),
                 re.compile(r'\b(sometimes|occasionally|periodically|rarely|infrequently|not always|not constantly|not invariably|not absolutely|sporadically|intermittently)\b', re.IGNORECASE)),
                
                # Никогда/возможно противоречия
                (re.compile(r'\b(never|not ever|impossible|no way|absolutely not|under no circumstances|by no means|in no way|not at all)\b', re.IGNORECASE),
                 re.compile(r'\b(sometimes|occasionally|periodically|rarely|infrequently|possible|sometimes possible|might|could|perhaps)\b', re.IGNORECASE)),
                
                # Истина/ложь противоречия
                (re.compile(r'\b(true|correct|accurate|valid|right|precise|exact|factual|verified|confirmed|proven)\b', re.IGNORECASE),
                 re.compile(r'\b(false|wrong|incorrect|invalid|inaccurate|imprecise|inexact|unfactual|unverified|disproven|refuted)\b', re.IGNORECASE)),
                
                # Хорошо/плохо противоречия
                (re.compile(r'\b(good|positive|beneficial|advantageous|excellent|superior|better|favorable|helpful|useful|effective)\b', re.IGNORECASE),
                 re.compile(r'\b(bad|negative|harmful|detrimental|poor|inferior|worse|unfavorable|harmful|useless|ineffective)\b', re.IGNORECASE)),
                
                # Существует/отсутствует противоречия
                (re.compile(r'\b(exists|present|available|there is|there are|occurs|happens|appears|manifests|shows up|is found|is detected)\b', re.IGNORECASE),
                 re.compile(r'\b(absent|missing|unavailable|lacking|nonexistent|gone|does not occur|does not happen|vanishes|disappears|is not found|is not detected)\b', re.IGNORECASE)),
                
                # Увеличивается/уменьшается противоречия
                (re.compile(r'\b(increases|grows|rises|expands|improves|enhances|accelerates|strengthens|builds up|gets bigger|gets stronger)\b', re.IGNORECASE),
                 re.compile(r'\b(decreases|shrinks|falls|declines|reduces|diminishes|decelerates|weakens|dies down|gets smaller|gets weaker)\b', re.IGNORECASE)),
                
                # Постоянный/изменчивый противоречия
                (re.compile(r'\b(constant|unchanging|stable|fixed|permanent|steady|consistent|unchangeable|immutable|invariable)\b', re.IGNORECASE),
                 re.compile(r'\b(changing|variable|fluctuating|unstable|temporary|inconsistent|inconstant|mutable|variable|shifting)\b', re.IGNORECASE)),
                
                # Абсолютный/частичный противоречия
                (re.compile(r'\b(absolutely|completely|totally|entirely|fully|100%|wholly|thoroughly|utterly|perfectly)\b', re.IGNORECASE),
                 re.compile(r'\b(partially|somewhat|partly|incompletely|to some extent|not fully|not completely|in part|incomplete|partial)\b', re.IGNORECASE)),
                
                # Определенный/неопределенный противоречия
                (re.compile(r'\b(certain|definite|sure|guaranteed|without doubt|unquestionable|undeniable|indisputable|incontrovertible|beyond doubt)\b', re.IGNORECASE),
                 re.compile(r'\b(uncertain|unclear|doubtful|questionable|possible|maybe|not certain|ambiguous|vague|indefinite)\b', re.IGNORECASE)),
                
                # Успешный/неудачный противоречия
                (re.compile(r'\b(successful|effective|working|functional|operational|productive|fruitful|profitable|achieving|accomplishing)\b', re.IGNORECASE),
                 re.compile(r'\b(unsuccessful|ineffective|failing|broken|malfunctioning|non-operational|unproductive|fruitless|unprofitable|failing|flawed)\b', re.IGNORECASE)),
                
                # Доказывает/опровергает противоречия
                (re.compile(r'\b(proves|demonstrates|confirms|establishes|shows|verifies|validates|substantiates|corroborates|attests)\b', re.IGNORECASE),
                 re.compile(r'\b(disproves|contradicts|refutes|challenges|questions|invalidates|undermines|debunks|discredits|negates)\b', re.IGNORECASE)),
                
                # Все/некоторые противоречия
                (re.compile(r'\b(all|every|each|100%|entire|complete|total|whole|every single|each and every)\b', re.IGNORECASE),
                 re.compile(r'\b(some|few|many|several|most|not all|not every|partial|a few|a number of|several|various)\b', re.IGNORECASE)),
                
                # Никто/кто-то противоречия
                (re.compile(r'\b(none|zero|no one|nobody|not any|absolutely none|not a single|not one person|not anybody)\b', re.IGNORECASE),
                 re.compile(r'\b(some|a few|several|many|at least one|someone|somebody|a number of|various people)\b', re.IGNORECASE)),
                
                # Большинство/меньшинство противоречия
                (re.compile(r'\b(majority|most|>50%|greater part|bulk|largest portion|predominant|prevalent)\b', re.IGNORECASE),
                 re.compile(r'\b(minority|few|<50%|smaller part|small portion|less common|less frequent|less prevalent)\b', re.IGNORECASE))
            ],
            
            'self_contradiction_keywords': [
                re.compile(r'\b(but|however|although|though|despite|nevertheless|nonetheless|yet|still|on the other hand|in contrast|whereas|while|even though|even if|notwithstanding|conversely|in spite of|regardless of|on the contrary|all the same|by contrast|in fact|as a matter of fact|at the same time|simultaneously)\b', re.IGNORECASE)
            ],
            
            'hidden_contradiction_patterns': [
                # Противоположные позиции
                (re.compile(r'\b(against|opposed to|disagree with|reject|oppose|contradict|refute|deny|dispute|object to|resist)\b', re.IGNORECASE),
                 re.compile(r'\b(for|in favor of|support|agree with|accept|endorse|approve|back|champion|defend|promote)\b', re.IGNORECASE)),
                
                # Любовь/ненависть противоречия
                (re.compile(r'\b(love|like|enjoy|appreciate|admire|value|cherish|treasure|adore|fond of|care for)\b', re.IGNORECASE),
                 re.compile(r'\b(hate|dislike|detest|despise|abhor|reject|loathe|can\'t stand|disapprove of|resent)\b', re.IGNORECASE)),
                
                # Согласие/несогласие противоречия
                (re.compile(r'\b(agree|support|approve|endorse|accept|approve of|concur with|side with|back up|stand by)\b', re.IGNORECASE),
                 re.compile(r'\b(disagree|oppose|reject|disapprove|oppose to|object to|refute|deny|contradict|argue against)\b', re.IGNORECASE)),
                
                # Включение/исключение противоречия
                (re.compile(r'\b(include|contain|have|possess|comprise|consist of|encompass|incorporate|feature|involve|cover)\b', re.IGNORECASE),
                 re.compile(r'\b(exclude|omit|lack|not have|absent|missing|do not include|leave out|omit|exclude from|rule out)\b', re.IGNORECASE)),
                
                # Помощь/вред противоречия
                (re.compile(r'\b(help|assist|aid|benefit|support|improve|enhance|promote|facilitate|advance)\b', re.IGNORECASE),
                 re.compile(r'\b(harm|hurt|damage|injure|worsen|impair|deteriorate|undermine|obstruct|hinder)\b', re.IGNORECASE)),
                
                # Начало/конец противоречия
                (re.compile(r'\b(begin|start|commence|initiate|launch|open|inaugurate|originate|establish)\b', re.IGNORECASE),
                 re.compile(r'\b(end|finish|conclude|terminate|close|stop|cease|complete|finalize|wrap up)\b', re.IGNORECASE))
            ],
            
            'temporal_contradiction_patterns': [
                # До/после противоречия
                (re.compile(r'\b(before|prior to|earlier than|previously|previously to|in advance of|preceding|earlier|sooner)\b', re.IGNORECASE),
                 re.compile(r'\b(after|following|later than|subsequently|afterwards|subsequently to|following|later|subsequent)\b', re.IGNORECASE)),
                
                # Первый/последний противоречия
                (re.compile(r'\b(first|initially|at first|in the beginning|at the outset|in the early stages|originally|to begin with|starting with)\b', re.IGNORECASE),
                 re.compile(r'\b(last|finally|ultimately|eventually|in the end|at the conclusion|in the final stages|ultimately|ending with|concluding with)\b', re.IGNORECASE)),
                
                # Немедленно/постепенно противоречия
                (re.compile(r'\b(immediately|right away|instantly|at once|without delay|promptly|straight away|directly|forthwith|instantaneously)\b', re.IGNORECASE),
                 re.compile(r'\b(eventually|gradually|over time|in the long run|after a while|in due course|slowly|progressively|step by step|little by little)\b', re.IGNORECASE)),
                
                # Постоянно/временно противоречия
                (re.compile(r'\b(permanently|forever|always|constantly|continuously|indefinitely|endlessly|unceasingly|eternally)\b', re.IGNORECASE),
                 re.compile(r'\b(temporarily|briefly|momentarily|for a while|short-term|transiently|ephemerally|for the time being|in the meantime)\b', re.IGNORECASE))
            ],
            
            'quantitative_contradiction_patterns': [
                # Все/некоторые противоречия
                (re.compile(r'\b(all|every|each|100%|entire|complete|total|whole|every single|each and every|the entirety of)\b', re.IGNORECASE),
                 re.compile(r'\b(some|few|many|several|most|not all|not every|partial|a few|a number of|several|various|a portion of)\b', re.IGNORECASE)),
                
                # Никто/кто-то противоречия
                (re.compile(r'\b(none|zero|no one|nobody|not any|absolutely none|not a single|not one person|not anybody|not a soul)\b', re.IGNORECASE),
                 re.compile(r'\b(some|a few|several|many|at least one|someone|somebody|a number of|various people|a handful of)\b', re.IGNORECASE)),
                
                # Большинство/меньшинство противоречия
                (re.compile(r'\b(majority|most|>50%|greater part|bulk|largest portion|predominant|prevalent|the larger part)\b', re.IGNORECASE),
                 re.compile(r'\b(minority|few|<50%|smaller part|small portion|less common|less frequent|less prevalent|the smaller part)\b', re.IGNORECASE)),
                
                # Всегда/иногда противоречия
                (re.compile(r'\b(always|constantly|permanently|without exception|invariably|every time|each time|on all occasions|at all times)\b', re.IGNORECASE),
                 re.compile(r'\b(sometimes|occasionally|rarely|infrequently|now and then|from time to time|every so often|once in a while)\b', re.IGNORECASE)),
                
                # Никогда/иногда противоречия
                (re.compile(r'\b(never|not ever|impossible|no way|absolutely not|under no circumstances|by no means|in no way|not at all|not once)\b', re.IGNORECASE),
                 re.compile(r'\b(sometimes|occasionally|rarely|infrequently|now and then|from time to time|every so often|once in a while|on occasion)\b', re.IGNORECASE)),
                
                # Много/мало противоречия
                (re.compile(r'\b(many|numerous|multiple|several|a lot of|plenty of|abundant|copious|a great deal of|a large number of)\b', re.IGNORECASE),
                 re.compile(r'\b(few|several|a handful of|a small number of|a couple of|minimal|scarce|limited|insufficient|inadequate)\b', re.IGNORECASE))
            ]
        }
    
    def _init_cache(self):
        """Инициализирует кэш для ускорения обработки"""
        self._contradiction_cache = {}
        self._sentence_cache = {}
        self._pattern_cache = {}
    
    def _find_direct_contradictions(self, text: str) -> List[Dict]:
        """
        Находит прямые противоречия через сопоставление паттернов.
        
        Аргументы:
            text (str): Текст для анализа
            
        Возвращает:
            List[Dict]: Список найденных противоречий
        """
        contradictions = []
        
        # Разделяем на предложения
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Проверяем каждое предложение на противоречия
        for i, sentence in enumerate(sentences):
            for pattern1, pattern2 in self.patterns['direct_contradictions']:
                try:
                    # Проверяем, присутствуют ли оба паттерна в предложении
                    if pattern1.search(sentence) and pattern2.search(sentence):
                        # Получаем точные совпадающие слова
                        match1 = pattern1.search(sentence)
                        match2 = pattern2.search(sentence)
                        
                        # Убеждаемся, что слова не в одной и той же фразе
                        if not self._is_same_phrase(match1, match2, sentence):
                            contradictions.append({
                                'type': 'direct_contradiction',
                                'sentence': sentence,
                                'index': i,
                                'pattern': (pattern1.pattern, pattern2.pattern),
                                'matched_words': (match1.group(0) if match1 else None, 
                                                 match2.group(0) if match2 else None),
                                'method': 'pattern_matching',
                                'confidence': 0.95
                            })
                except Exception as e:
                    # Пропускаем ошибки в отдельных паттернах
                    continue
        
        return contradictions
    
    def _find_self_contradictions(self, text: str) -> List[Dict]:
        """
        Находит самопротиворечия через ключевые слова с анализом контекста.
        
        Аргументы:
            text (str): Текст для анализа
            
        Возвращает:
            List[Dict]: Список найденных самопротиворечий
        """
        contradictions = []
        
        # Проверяем каждое ключевое слово самопротиворечия
        for keyword_pattern in self.patterns['self_contradiction_keywords']:
            try:
                matches = list(keyword_pattern.finditer(text))
                
                for match in matches:
                    # Извлекаем контекст (50 символов до и после)
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    # Проверяем, есть ли противоречие в контексте
                    if self._has_contradiction_in_context(context):
                        contradictions.append({
                            'type': 'self_contradiction',
                            'keyword': match.group(0),
                            'context': context,
                            'position': match.start(),
                            'method': 'contextual_analysis',
                            'confidence': 0.75
                        })
            except Exception as e:
                # Пропускаем ошибки
                continue
        
        return contradictions
    
    def _has_contradiction_in_context(self, context: str) -> bool:
        """
        Проверяет, содержит ли контекст противоречие.
        
        Аргументы:
            context (str): Контекст для проверки
            
        Возвращает:
            bool: True если найдено противоречие
        """
        # Проверяем прямые противоречия в контексте
        for pattern1, pattern2 in self.patterns['direct_contradictions']:
            try:
                if pattern1.search(context) and pattern2.search(context):
                    return True
            except Exception as e:
                continue
        
        # Проверяем скрытые противоречия в контексте
        for pattern1, pattern2 in self.patterns['hidden_contradiction_patterns']:
            try:
                if pattern1.search(context) and pattern2.search(context):
                    return True
            except Exception as e:
                continue
        
        return False
    
    def _is_same_phrase(self, match1, match2, sentence: str) -> bool:
        """
        Проверяет, находятся ли два совпадения в одной и той же фразе.
        
        Аргументы:
            match1: Первое совпадение
            match2: Второе совпадение
            sentence (str): Предложение
            
        Возвращает:
            bool: True если в одной фразе
        """
        # Проверяем, находятся ли совпадения в пределах 15 символов друг от друга
        if abs(match1.start() - match2.start()) < 15:
            return True
        
        # Проверяем, находятся ли совпадения в одном предложении
        if re.search(r'\b(and|but|or|however|although|though|while|yet|nevertheless|nonetheless)\b', 
                    sentence[min(match1.start(), match2.start()):max(match1.end(), match2.end())]):
            return True
        
        return False
    
    def _find_hidden_contradictions(self, text: str) -> List[Dict]:
        """
        Находит скрытые противоречия с семантическим анализом.
        
        Аргументы:
            text (str): Текст для анализа
            
        Возвращает:
            List[Dict]: Список найденных скрытых противоречий
        """
        contradictions = []
        
        # Проверяем каждый паттерн
        for pattern1, pattern2 in self.patterns['hidden_contradiction_patterns']:
            try:
                matches1 = pattern1.finditer(text)
                matches2 = pattern2.finditer(text)
                
                # Создаем список всех позиций совпадений
                match_positions = []
                for match in matches1:
                    match_positions.append((match.start(), match.end(), 'pattern1'))
                for match in matches2:
                    match_positions.append((match.start(), match.end(), 'pattern2'))
                
                # Сортируем по позиции
                match_positions.sort(key=lambda x: x[0])
                
                # Проверяем близкие совпадения (в пределах 50 символов)
                for i in range(len(match_positions)):
                    for j in range(i+1, len(match_positions)):
                        pos1, pos2 = match_positions[i], match_positions[j]
                        
                        # Если совпадения близки друг к другу
                        if abs(pos1[0] - pos2[0]) < 50 and pos1[2] != pos2[2]:
                            # Получаем контекст вокруг совпадений
                            start = max(0, min(pos1[0], pos2[0]) - 30)
                            end = min(len(text), max(pos1[1], pos2[1]) + 30)
                            context = text[start:end]
                            
                            contradictions.append({
                                'type': 'hidden_contradiction',
                                'context': context,
                                'position': min(pos1[0], pos2[0]),
                                'matched_pattern': f"{pos1[2]} vs {pos2[2]}",
                                'method': 'pattern_proximity',
                                'confidence': 0.8
                            })
            except Exception as e:
                # Пропускаем ошибки
                continue
        
        return contradictions
    
    def _find_temporal_contradictions(self, text: str) -> List[Dict]:
        """
        Находит противоречия во временных утверждениях.
        
        Аргументы:
            text (str): Текст для анализа
            
        Возвращает:
            List[Dict]: Список найденных временных противоречий
        """
        contradictions = []
        
        # Разделяем на предложения
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Проверяем каждое предложение на временные противоречия
        for i, sentence in enumerate(sentences):
            for pattern1, pattern2 in self.patterns['temporal_contradiction_patterns']:
                try:
                    if pattern1.search(sentence) and pattern2.search(sentence):
                        contradictions.append({
                            'type': 'temporal_contradiction',
                            'sentence': sentence,
                            'index': i,
                            'pattern': (pattern1.pattern, pattern2.pattern),
                            'method': 'temporal_pattern',
                            'confidence': 0.88
                        })
                except Exception as e:
                    # Пропускаем ошибки
                    continue
        
        return contradictions
    
    def _find_quantitative_contradictions(self, text: str) -> List[Dict]:
        """
        Находит противоречия в количественных утверждениях.
        
        Аргументы:
            text (str): Текст для анализа
            
        Возвращает:
            List[Dict]: Список найденных количественных противоречий
        """
        contradictions = []
        
        # Разделяем на предложения
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Проверяем каждое предложение на количественные противоречия
        for i, sentence in enumerate(sentences):
            for pattern1, pattern2 in self.patterns['quantitative_contradiction_patterns']:
                try:
                    if pattern1.search(sentence) and pattern2.search(sentence):
                        contradictions.append({
                            'type': 'quantitative_contradiction',
                            'sentence': sentence,
                            'index': i,
                            'pattern': (pattern1.pattern, pattern2.pattern),
                            'method': 'quantitative_pattern',
                            'confidence': 0.85
                        })
                except Exception as e:
                    # Пропускаем ошибки
                    continue
        
        return contradictions
    
    def _semantic_analysis(self, text: str) -> List[Dict]:
        """
        Семантический анализ через модель NLI.
        
        Аргументы:
            text (str): Текст для анализа
            
        Возвращает:
            List[Dict]: Список найденных семантических противоречий
        """
        if not self.nli_pipeline:
            return []
        
        contradictions = []
        
        # Разделяем на предложения
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
        
        if len(sentences) < 2:
            return contradictions
        
        # Сравниваем каждую пару предложений
        for i in range(len(sentences) - 1):
            for j in range(i + 1, len(sentences)):
                premise = sentences[i]
                hypothesis = sentences[j]
                
                # Пропускаем слишком короткие предложения
                if len(premise) < 15 or len(hypothesis) < 15:
                    continue
                
                try:
                    # Используем модель NLI для проверки противоречия
                    result = self.nli_pipeline(
                        premise,
                        candidate_labels=["yes", "it is not possible to tell", "no"]
                    )
                    
                    # Проверяем, обнаружила ли модель противоречие с высокой уверенностью
                    contradiction_score = 0
                    for label, score in zip(result['labels'], result['scores']):
                        if label == 'no':  # 'no' означает противоречие в NLI
                            contradiction_score = score
                            break
                    
                    # Если уверенность высокая (> 0.75)
                    if contradiction_score > 0.75:
                        contradictions.append({
                            'type': 'semantic_contradiction',
                            'premise': premise,
                            'hypothesis': hypothesis,
                            'confidence': round(contradiction_score, 3),
                            'method': 'semantic_nli',
                            'model_score': contradiction_score,
                            'impact': round(contradiction_score * 1.8, 3)
                        })
                except Exception as e:
                    # Пропускаем ошибки модели
                    continue
        
        return contradictions
    
    def detect_contradictions(self, text: str) -> List[Dict]:
        """
        Основной метод обнаружения противоречий.
        
        Аргументы:
            text (str): Текст для анализа
            
        Возвращает:
            List[Dict]: Список всех найденных противоречий
        """
        if not text or not isinstance(text, str):
            raise ValueError("Текст должен быть непустой строкой")
        
        contradictions = []
        
        # 1. Прямые противоречия через паттерны
        direct_contradictions = self._find_direct_contradictions(text)
        contradictions.extend(direct_contradictions)
        
        # 2. Самопротиворечия через ключевые слова
        self_contradictions = self._find_self_contradictions(text)
        contradictions.extend(self_contradictions)
        
        # 3. Скрытые противоречия через паттерны
        hidden_contradictions = self._find_hidden_contradictions(text)
        contradictions.extend(hidden_contradictions)
        
        # 4. Временные противоречия
        temporal_contradictions = self._find_temporal_contradictions(text)
        contradictions.extend(temporal_contradictions)
        
        # 5. Количественные противоречия
        quantitative_contradictions = self._find_quantitative_contradictions(text)
        contradictions.extend(quantitative_contradictions)
        
        # 6. Семантический анализ через модель NLI (если доступна)
        if self.nli_pipeline:
            semantic_contradictions = self._semantic_analysis(text)
            contradictions.extend(semantic_contradictions)
        
        # Удаляем дубликаты
        contradictions = self._remove_duplicates(contradictions)
        
        # Добавляем оценки уверенности и серьёзности
        contradictions = self._add_confidence_scores(contradictions, text)
        
        return contradictions
    
    def _remove_duplicates(self, contradictions: List[Dict]) -> List[Dict]:
        """
        Удаляет дубликаты противоречий с проверкой на схожесть.
        
        Аргументы:
            contradictions (List[Dict]): Список противоречий
            
        Возвращает:
            List[Dict]: Уникальные противоречия
        """
        unique = []
        seen = set()
        
        for c in contradictions:
            # Создаем уникальный ключ на основе типа и содержания
            key = (
                c.get('type'),
                c.get('sentence', '')[:40] if c.get('sentence') else '',
                c.get('context', '')[:40] if c.get('context') else '',
                c.get('keyword', '')
            )
            
            # Проверяем на похожие противоречия
            is_duplicate = False
            for existing in unique:
                if self._is_similar(c, existing):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen.add(key)
                unique.append(c)
        
        return unique
    
    def _is_similar(self, c1: Dict, c2: Dict) -> bool:
        """
        Проверяет, похожи ли два противоречия.
        
        Аргументы:
            c1 (Dict): Первое противоречие
            c2 (Dict): Второе противоречие
            
        Возвращает:
            bool: True если похожи
        """
        # Проверяем тип
        if c1['type'] != c2['type']:
            return False
        
        # Проверяем схожесть контекстов
        context1 = c1.get('context', '')
        context2 = c2.get('context', '')
        
        # Если оба имеют контексты, проверяем схожесть
        if context1 and context2:
            # Простая проверка схожести на основе общих слов
            words1 = set(re.findall(r'\w+', context1.lower()))
            words2 = set(re.findall(r'\w+', context2.lower()))
            
            common_words = words1 & words2
            similarity = len(common_words) / max(len(words1), len(words2), 1)
            
            return similarity > 0.75
        
        # Если одно имеет предложение, а другое контекст
        if (c1.get('sentence') and c2.get('context')) or (c1.get('context') and c2.get('sentence')):
            return False
        
        # Если оба имеют предложения, проверяем схожесть
        sentence1 = c1.get('sentence', '')
        sentence2 = c2.get('sentence', '')
        
        if sentence1 and sentence2:
            words1 = set(re.findall(r'\w+', sentence1.lower()))
            words2 = set(re.findall(r'\w+', sentence2.lower()))
            
            common_words = words1 & words2
            similarity = len(common_words) / max(len(words1), len(words2), 1)
            
            return similarity > 0.75
        
        return False
    
    def _add_confidence_scores(self, contradictions: List[Dict], text: str) -> List[Dict]:
        """
        Добавляет оценки уверенности и уровни серьёзности.
        
        Аргументы:
            contradictions (List[Dict]): Список противоречий
            text (str): Исходный текст
            
        Возвращает:
            List[Dict]: Противоречия с оценками
        """
        for c in contradictions:
            # Назначаем серьёзность на основе типа противоречия
            severity_map = {
                'direct_contradiction': ('high', 18),
                'semantic_contradiction': ('high', 15),
                'temporal_contradiction': ('medium', 10),
                'quantitative_contradiction': ('medium', 8),
                'self_contradiction': ('medium', 7),
                'hidden_contradiction': ('low', 5)
            }
            
            c_type = c['type']
            if c_type in severity_map:
                c['severity'] = severity_map[c_type][0]
                c['severity_score'] = severity_map[c_type][1]
            else:
                c['severity'] = 'low'
                c['severity_score'] = 4
            
            # Вычисляем влияние
            c['impact'] = round(c['severity_score'] * c['confidence'] * 1.8, 2)
        
        return contradictions
    
    def generate_report(self, contradictions: List[Dict], text_length: int) -> Dict:
        """
        Генерирует подробный отчёт.
        
        Аргументы:
            contradictions (List[Dict]): Список противоречий
            text_length (int): Длина текста
            
        Возвращает:
            Dict: Отчёт с детальной статистикой
        """
        if not contradictions:
            return {
                'summary': 'No contradictions found.',
                'total': 0,
                'by_type': {},
                'consistency_score': 100.0,
                'recommendation': 'Text is logically consistent.',
                'contradictions': [],
                'confidence_distribution': self._get_confidence_distribution(contradictions),
                'severity_distribution': self._get_severity_distribution(contradictions),
                'quality_metrics': self._calculate_quality_metrics(contradictions)
            }
        
        # Статистика по типам
        by_type = defaultdict(lambda: {'count': 0, 'total_impact': 0, 'examples': []})
        total_impact = 0
        total_confidence = 0
        
        for c in contradictions:
            c_type = c['type']
            by_type[c_type]['count'] += 1
            by_type[c_type]['total_impact'] += c['impact']
            
            # Сохраняем первые 3 примера на тип
            if len(by_type[c_type]['examples']) < 3:
                example = {
                    'sentence': c.get('sentence', c.get('context', ''))[:100],
                    'severity': c['severity'],
                    'confidence': c['confidence'],
                    'impact': c['impact']
                }
                by_type[c_type]['examples'].append(example)
            
            total_impact += c['impact']
            total_confidence += c['confidence']
        
        # Вычисляем оценку согласованности (0-100%)
        consistency_score = max(0.0, 100.0 - total_impact)
        
        # Генерируем рекомендацию
        recommendation = self._generate_recommendation(consistency_score)
        
        return {
            'summary': f'Found {len(contradictions)} contradiction(s)',
            'total': len(contradictions),
            'by_type': dict(by_type),
            'consistency_score': round(consistency_score, 1),
            'recommendation': recommendation,
            'contradictions': contradictions,
            'confidence_distribution': self._get_confidence_distribution(contradictions),
            'severity_distribution': self._get_severity_distribution(contradictions),
            'quality_metrics': self._calculate_quality_metrics(contradictions)
        }
    
    def _get_confidence_distribution(self, contradictions: List[Dict]) -> Dict:
        """Получает распределение оценок уверенности"""
        distribution = {
            'high': 0,  # > 0.8
            'medium': 0,  # 0.5-0.8
            'low': 0  # < 0.5
        }
        
        for c in contradictions:
            if c['confidence'] > 0.8:
                distribution['high'] += 1
            elif c['confidence'] > 0.5:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1
        
        return distribution
    
    def _get_severity_distribution(self, contradictions: List[Dict]) -> Dict:
        """Получает распределение уровней серьёзности"""
        distribution = {
            'high': 0,
            'medium': 0,
            'low': 0
        }
        
        for c in contradictions:
            distribution[c['severity']] += 1
        
        return distribution
    
    def _calculate_quality_metrics(self, contradictions: List[Dict]) -> Dict:
        """Вычисляет метрики качества текста"""
        total = len(contradictions)
        if total == 0:
            return {
                'consistency': 100.0,
                'reliability': 100.0,
                'coherence': 100.0,
                'logical_strength': 100.0
            }
        
        # Вычисляем метрики
        metrics = {
            'consistency': max(0, 100 - sum(c['impact'] for c in contradictions)),
            'reliability': max(0, 100 - (sum(c['confidence'] for c in contradictions) / total) * 20),
            'coherence': max(0, 100 - (sum(c['severity_score'] for c in contradictions) / total) * 5),
            'logical_strength': max(0, 100 - (sum(c['impact'] for c in contradictions) / total) * 3)
        }
        
        return {k: round(v, 1) for k, v in metrics.items()}
    
    def _generate_recommendation(self, consistency_score: float) -> str:
        """Генерирует рекомендацию на основе оценки согласованности"""
        if consistency_score >= 95:
            return "Excellent! Text is logically consistent with minimal issues."
        elif consistency_score >= 85:
            return "Very good. Text demonstrates strong logical coherence."
        elif consistency_score >= 75:
            return "Good. Minor logical inconsistencies present."
        elif consistency_score >= 65:
            return "Acceptable. Some contradictions need attention."
        elif consistency_score >= 55:
            return "Moderate. Significant logical issues detected."
        elif consistency_score >= 45:
            return "Poor. Text contains multiple serious contradictions."
        elif consistency_score >= 35:
            return "Very poor. Major logical flaws throughout the text."
        else:
            return "Critical. Text is riddled with contradictions and lacks logical coherence."