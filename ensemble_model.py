"""
Kazakh Morphology Ensemble System
==================================

This ensemble combines multiple specialized models:
1. Structure Model: Handles morphology and basic semantic classification
2. Lexical Model: Generates word meanings (lexics.мағынасы)
3. Sozjasam Model: Predicts word formation patterns with retrieval
4. Validator: Ensures JSON correctness and constraint compliance
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MorphologyOutput:
    """Container for morphology analysis output"""
    pos_tag: str
    word: str
    lemma: str
    morphology: Dict[str, str]
    semantics: Dict[str, str]
    lexics: Dict[str, str]
    sozjasam: Dict[str, str]
    confidence: float = 0.0
    source: str = "unknown"
    
    def to_dict(self) -> Dict:
        return {
            "POS tag": self.pos_tag,
            "word": self.word,
            "lemma": self.lemma,
            "morphology": self.morphology,
            "semantics": self.semantics,
            "lexics": self.lexics,
            "sozjasam": self.sozjasam
        }


class BaseModel(ABC):
    """Base class for all specialized models"""
    
    def __init__(self, model_name: str, model_path: Optional[str] = None):
        self.model_name = model_name
        self.model_path = model_path
        
    @abstractmethod
    def predict(self, word: str, pos_tag: str, context: Optional[Dict] = None) -> Dict:
        """Predict output for given word and POS tag"""
        pass
    
    @abstractmethod
    def get_confidence(self, output: Dict) -> float:
        """Calculate confidence score for output"""
        pass


class StructureModel(BaseModel):
    """
    Handles morphology structure and POS-specific semantics.
    This is the primary model for structural analysis.
    
    Best for:
    - morphology.column (41-100% accuracy)
    - morphology.дара, негізгі (76-100%)
    - morphology.дара, туынды (61-95%)
    - Basic semantic categories (70-100%)
    """
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-7B-Instruct"):
        super().__init__("StructureModel", model_path)
        self.model = None  # Will be loaded with transformers
        self.tokenizer = None
        
    def load_model(self):
        """Load the fine-tuned structure model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading structure model: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info("Structure model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load structure model: {e}")
            raise
    
    def predict(self, word: str, pos_tag: str, context: Optional[Dict] = None) -> Dict:
        """Predict morphology structure and semantics"""
        
        if self.model is None:
            self.load_model()
        
        # Build prompt focusing on structure
        prompt = self._build_structure_prompt(word, pos_tag, context)
        
        # Generate response
        output = self._generate(prompt)
        
        # Parse JSON
        try:
            result = json.loads(output)
            
            # Remove lexics and sozjasam from structure model output
            # These will be filled by specialized models
            if 'lexics' in result:
                result['lexics'] = {'мағынасы': 'NaN'}
            if 'sozjasam' in result:
                result['sozjasam'] = {'тәсілін, құрамын шартты қысқартумен беру': 'NaN'}
            
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error in structure model: {e}")
            return self._get_default_structure(word, pos_tag)
    
    def _build_structure_prompt(self, word: str, pos_tag: str, context: Optional[Dict]) -> str:
        """Build prompt for structure prediction"""
        
        prompt = f"""Сіз қазақ тілінің морфологиясы мен семантикасы бойынша сарапшысыз.

МІНДЕТ: Берілген сөздің МОРФОЛОГИЯЛЫҚ ҚҰРЫЛЫМЫН және СЕМАНТИКАЛЫҚ КАТЕГОРИЯСЫН анықтаңыз.

СӨЗ: {word}
СӨЗ ТАБЫ: {pos_tag}

НҰСҚАУЛАР:
1. ТЕК JSON форматында жауап беріңіз
2. Морфология бөлімін толық толтырыңыз (column, дара/күрделі, негізгі/туынды)
3. Семантика бөлімін POS табына сәйкес толтырыңыз
4. lemma анықтаңыз
5. lexics және sozjasam үшін "NaN" қойыңыз (олар басқа модельдермен толтырылады)

JSON ҚҰРЫЛЫМЫ:
{{
  "POS tag": "{pos_tag}",
  "word": "{word}",
  "lemma": "...",
  "morphology": {{
    "column": "...",
    "дара, негізгі": "...",
    "дара, туынды": "...",
    "күрделі, біріккен, Бірік.": "...",
    "күрделі, қосарланған, Қос.": "...",
    "күрделі, қысқарған, Қыс.": "...",
    "күрделі, тіркескен, Тірк.": "..."
  }},
  "semantics": {{...}},
  "lexics": {{"мағынасы": "NaN"}},
  "sozjasam": {{"тәсілін, құрамын шартты қысқартумен беру": "NaN"}}
}}

МАҢЫЗДЫ: Тек қана валидті JSON шығарыңыз. Басқа мәтін жоқ."""

        return prompt
    
    def _generate(self, prompt: str) -> str:
        """Generate output using the model"""
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,  # Low temperature for consistency
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "{" in response and "}" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            response = response[start:end]
        
        return response
    
    def _get_default_structure(self, word: str, pos_tag: str) -> Dict:
        """Return default structure when parsing fails"""
        return {
            "POS tag": pos_tag,
            "word": word,
            "lemma": word,
            "morphology": {
                "column": "NaN",
                "дара, негізгі": "NaN",
                "дара, туынды": "NaN",
                "күрделі, біріккен, Бірік.": "NaN",
                "күрделі, қосарланған, Қос.": "NaN",
                "күрделі, қысқарған, Қыс.": "NaN",
                "күрделі, тіркескен, Тірк.": "NaN"
            },
            "semantics": {},
            "lexics": {"мағынасы": "NaN"},
            "sozjasam": {"тәсілін, құрамын шартты қысқартумен беру": "NaN"}
        }
    
    def get_confidence(self, output: Dict) -> float:
        """Calculate confidence based on completeness"""
        score = 0.0
        total = 0
        
        # Check morphology completeness
        if 'morphology' in output:
            for key, value in output['morphology'].items():
                if key != 'column':
                    total += 1
                    if value != 'NaN':
                        score += 1
        
        # Check semantics completeness
        if 'semantics' in output:
            total += 1
            has_non_nan = any(v != 'NaN' for v in output['semantics'].values())
            if has_non_nan:
                score += 1
        
        return score / total if total > 0 else 0.5


class LexicalModel(BaseModel):
    """
    Specialized model for generating word meanings (lexics.мағынасы).
    Uses larger model with better language understanding.
    
    Target: lexics.мағынасы (8-48% accuracy → goal: 60-75%)
    """
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-14B-Instruct"):
        super().__init__("LexicalModel", model_path)
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load the lexical model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading lexical model: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True  # Use 8-bit quantization for 14B model
            )
            logger.info("Lexical model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load lexical model: {e}")
            raise
    
    def predict(self, word: str, pos_tag: str, context: Optional[Dict] = None) -> Dict:
        """Generate word meaning"""
        
        if self.model is None:
            self.load_model()
        
        # Build prompt with morphological context
        prompt = self._build_lexical_prompt(word, pos_tag, context)
        
        # Generate meaning
        meaning = self._generate(prompt)
        
        return {"мағынасы": meaning}
    
    def _build_lexical_prompt(self, word: str, pos_tag: str, context: Optional[Dict]) -> str:
        """Build prompt for meaning generation"""
        
        morph_info = ""
        if context and 'morphology' in context:
            column = context['morphology'].get('column', 'NaN')
            if column != 'NaN':
                morph_info = f"\nМорфологиялық форма: {column}"
        
        sem_info = ""
        if context and 'semantics' in context:
            active_sem = [k for k, v in context['semantics'].items() if v != 'NaN']
            if active_sem:
                sem_info = f"\nСемантикалық категория: {', '.join(active_sem)}"
        
        prompt = f"""Сіз қазақ тілінің лексикологиясы бойынша сарапшысыз.

МІНДЕТ: Берілген сөздің ТОЛЫҚ ЛЕКСИКАЛЫҚ МАҒЫНАСЫН беріңіз.

СӨЗ: {word}
СӨЗ ТАБЫ: {pos_tag}{morph_info}{sem_info}

НҰСҚАУЛАР:
1. Сөздің негізгі мағынасын толық және нақты түсіндіріңіз
2. Сөз табын атап өтіңіз ({pos_tag})
3. Қазақ тілінде жазыңыз
4. Мысалдар қосуға болады
5. ТЕК мағынаны жазыңыз, JSON немесе басқа форматсыз

МЫСАЛДАР:
- "кітап": "кітап -зт. Басылып шыққан еңбек; мағлұматтарды, ойларды жинақтаған материалдық зат."
- "жылдам": "жылдам -сн. Қозғалысы жылдам, тез әрекет ететін, уақыт жағынан аз."

Мағынасы:"""

        return prompt
    
    def _generate(self, prompt: str) -> str:
        """Generate meaning using the model"""
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.95,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the meaning from response
        if "Мағынасы:" in response:
            meaning = response.split("Мағынасы:")[-1].strip()
        else:
            meaning = response.split(prompt)[-1].strip()
        
        # Clean up the meaning
        meaning = meaning.strip('"').strip("'").strip()
        
        # Ensure it starts with word and POS tag
        if not meaning.startswith(f"{word}"):
            meaning = f"{word} -{self._get_pos_abbrev(pos_tag)}. {meaning}"
        
        return meaning if meaning else "NaN"
    
    def _get_pos_abbrev(self, pos_tag: str) -> str:
        """Get POS tag abbreviation"""
        abbrevs = {
            'Зат есім': 'зт',
            'Сын есім': 'сн',
            'Етістік': 'ет',
            'Үстеу': 'үст',
            'Сан есім': 'сан',
            'Есімдік': 'ес',
            'Еліктеуіш': 'елс',
            'Одағай': 'од',
            'Шылау': 'шыл'
        }
        return abbrevs.get(pos_tag, 'сөз')
    
    def get_confidence(self, output: Dict) -> float:
        """Calculate confidence based on meaning quality"""
        meaning = output.get('мағынасы', '')
        
        if meaning == 'NaN' or not meaning:
            return 0.0
        
        # Check quality indicators
        score = 0.5  # Base score
        
        # Has reasonable length
        if len(meaning) > 20:
            score += 0.2
        
        # Contains POS tag indicator
        if any(tag in meaning for tag in ['-зт.', '-сн.', '-ет.', '-үст.', '-сан.']):
            score += 0.2
        
        # Properly formatted
        if '.' in meaning:
            score += 0.1
        
        return min(score, 1.0)


class SozjasamModel(BaseModel):
    """
    Word formation pattern predictor using retrieval-augmented generation.
    
    Target: sozjasam (23-83% accuracy → goal: 70-85%)
    """
    
    def __init__(self, model_path: str = "microsoft/Phi-3-mini-4k-instruct", 
                 pattern_db_path: Optional[str] = None):
        super().__init__("SozjasamModel", model_path)
        self.model = None
        self.tokenizer = None
        self.pattern_db = None
        self.pattern_db_path = pattern_db_path
        
    def load_model(self):
        """Load the sozjasam model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading sozjasam model: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Sozjasam model loaded successfully")
            
            # Load pattern database
            if self.pattern_db_path:
                self._load_pattern_db()
        except Exception as e:
            logger.error(f"Failed to load sozjasam model: {e}")
            raise
    
    def _load_pattern_db(self):
        """Load pattern database for retrieval"""
        try:
            with open(self.pattern_db_path, 'r', encoding='utf-8') as f:
                self.pattern_db = json.load(f)
            logger.info(f"Loaded {len(self.pattern_db)} pattern examples")
        except Exception as e:
            logger.warning(f"Could not load pattern database: {e}")
            self.pattern_db = {}
    
    def predict(self, word: str, pos_tag: str, context: Optional[Dict] = None) -> Dict:
        """Predict word formation pattern"""
        
        if self.model is None:
            self.load_model()
        
        # Retrieve similar examples
        similar_patterns = self._retrieve_similar_patterns(word, pos_tag, context)
        
        # Build prompt with examples
        prompt = self._build_sozjasam_prompt(word, pos_tag, context, similar_patterns)
        
        # Generate pattern
        pattern = self._generate(prompt)
        
        return {"тәсілін, құрамын шартты қысқартумен беру": pattern}
    
    def _retrieve_similar_patterns(self, word: str, pos_tag: str, 
                                   context: Optional[Dict], k: int = 3) -> List[Dict]:
        """Retrieve similar word patterns"""
        
        if not self.pattern_db:
            return []
        
        # Filter by POS tag
        same_pos = [p for p in self.pattern_db if p.get('POS tag') == pos_tag]
        
        if not same_pos:
            return []
        
        # Get morphology-based similarity
        if context and 'morphology' in context:
            morph_type = self._get_morphology_type(context['morphology'])
            
            # Filter by morphology type
            filtered = [p for p in same_pos 
                       if self._get_morphology_type(p.get('morphology', {})) == morph_type]
            
            if filtered:
                same_pos = filtered
        
        # Return top k examples
        return same_pos[:k]
    
    def _get_morphology_type(self, morphology: Dict) -> str:
        """Determine morphology type (дара/күрделі, негізгі/туынды)"""
        
        if morphology.get('дара, негізгі') != 'NaN':
            return 'дара_негізгі'
        elif morphology.get('дара, туынды') != 'NaN':
            return 'дара_туынды'
        elif morphology.get('күрделі, біріккен, Бірік.') != 'NaN':
            return 'күрделі_біріккен'
        elif morphology.get('күрделі, қосарланған, Қос.') != 'NaN':
            return 'күрделі_қосарланған'
        else:
            return 'unknown'
    
    def _build_sozjasam_prompt(self, word: str, pos_tag: str, 
                               context: Optional[Dict], 
                               similar_patterns: List[Dict]) -> str:
        """Build prompt with retrieval examples"""
        
        examples_str = ""
        if similar_patterns:
            examples_str = "\n\nҰҚСАС МЫСАЛДАР:"
            for i, ex in enumerate(similar_patterns, 1):
                ex_word = ex.get('word', '')
                ex_pattern = ex.get('sozjasam', {}).get('тәсілін, құрамын шартты қысқартумен беру', '')
                examples_str += f"\n{i}. {ex_word}: {ex_pattern}"
        
        morph_info = ""
        if context and 'morphology' in context:
            morph_type = self._get_morphology_type(context['morphology'])
            morph_info = f"\nМорфология түрі: {morph_type}"
        
        prompt = f"""Сіз қазақ тілінің сөзжасамы бойынша сарапшысыз.

МІНДЕТ: Берілген сөздің СӨЗЖАСАМ ТӘСІЛІН қысқартумен беріңіз.

СӨЗ: {word}
СӨЗ ТАБЫ: {pos_tag}{morph_info}{examples_str}

ҚЫСҚАРТУ ФОРМАТТАРЫ:
- Дара, негізгі: зт/Ø, сн/Ø
- Дара, туынды: зт/-шы, ет/-ла
- Күрделі, біріккен: зт+зт, сн+зт
- Күрделі, қосарланған: зт-зт

НҰСҚАУЛАР:
1. ТЕК қысқартылған форматты жазыңыз
2. Ұқсас мысалдарды пайдаланыңыз
3. Басқа түсініктеме жоқ

Қысқарту:"""

        return prompt
    
    def _generate(self, prompt: str) -> str:
        """Generate pattern using the model"""
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.1,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the pattern
        if "Қысқарту:" in response:
            pattern = response.split("Қысқарту:")[-1].strip()
        else:
            pattern = response.split(prompt)[-1].strip()
        
        # Clean up
        pattern = pattern.strip().split('\n')[0].strip()
        
        # Validate pattern format
        if not self._is_valid_pattern(pattern):
            return "NaN"
        
        return pattern
    
    def _is_valid_pattern(self, pattern: str) -> bool:
        """Check if pattern is valid"""
        if not pattern or pattern == "NaN":
            return False
        
        # Check for common pattern indicators
        valid_chars = ['/', '-', '+', 'Ø']
        has_valid_char = any(c in pattern for c in valid_chars)
        
        # Check for POS abbreviations
        pos_abbrevs = ['зт', 'сн', 'ет', 'үст', 'сан', 'ес']
        has_pos = any(abbrev in pattern for abbrev in pos_abbrevs)
        
        return has_valid_char or has_pos
    
    def get_confidence(self, output: Dict) -> float:
        """Calculate confidence based on pattern validity"""
        pattern = output.get('тәсілін, құрамын шартты қысқартумен беру', '')
        
        if not self._is_valid_pattern(pattern):
            return 0.0
        
        # Higher confidence for well-formatted patterns
        score = 0.6
        
        if '/' in pattern:
            score += 0.2
        if any(c in pattern for c in ['+', '-']):
            score += 0.1
        if any(abbrev in pattern for abbrev in ['зт', 'сн', 'ет']):
            score += 0.1
        
        return min(score, 1.0)


class OutputValidator:
    """
    Validates and fixes morphology outputs.
    Ensures JSON correctness and constraint compliance.
    """
    
    def __init__(self, grammar_data: Dict):
        self.grammar_data = grammar_data
        self.pos_structures = self._build_pos_structures()
    
    def _build_pos_structures(self) -> Dict:
        """Build expected structures for each POS tag"""
        structures = {}
        
        for pos_tag, data in self.grammar_data.items():
            structures[pos_tag] = {
                'morphology_keys': [],
                'semantic_keys': [],
                'rules': []
            }
            
            # Extract expected keys from grammar data
            if isinstance(data, dict):
                for key, value in data.items():
                    if 'morphology' in key.lower() or key == 'column':
                        structures[pos_tag]['morphology_keys'].append(key)
                    elif 'semantics' in key.lower() or any(sem in key for sem in ['сн', 'зт', 'ет']):
                        structures[pos_tag]['semantic_keys'].append(key)
        
        return structures
    
    def validate(self, output: Dict, pos_tag: str) -> Tuple[bool, List[str]]:
        """Validate output and return (is_valid, errors)"""
        errors = []
        
        # Check required keys
        required_keys = ['POS tag', 'word', 'lemma', 'morphology', 'semantics', 'lexics', 'sozjasam']
        for key in required_keys:
            if key not in output:
                errors.append(f"Missing required key: {key}")
        
        # Validate morphology
        if 'morphology' in output:
            morph_errors = self._validate_morphology(output['morphology'])
            errors.extend(morph_errors)
        
        # Validate semantics
        if 'semantics' in output:
            sem_errors = self._validate_semantics(output['semantics'], pos_tag)
            errors.extend(sem_errors)
        
        return len(errors) == 0, errors
    
    def _validate_morphology(self, morphology: Dict) -> List[str]:
        """Validate morphology structure"""
        errors = []
        
        # Rule 1: Exactly one of (дара, негізгі) or (дара, туынды) must be non-NaN
        dara_negizgi = morphology.get('дара, негізгі', 'NaN')
        dara_tuyndy = morphology.get('дара, туынды', 'NaN')
        
        dara_count = sum([dara_negizgi != 'NaN', dara_tuyndy != 'NaN'])
        
        # Rule 2: If күрделі is set, дара must be NaN
        kurdelі_count = sum([
            morphology.get('күрделі, біріккен, Бірік.', 'NaN') != 'NaN',
            morphology.get('күрделі, қосарланған, Қос.', 'NaN') != 'NaN',
            morphology.get('күрделі, қысқарған, Қыс.', 'NaN') != 'NaN',
            morphology.get('күрделі, тіркескен, Тірк.', 'NaN') != 'NaN'
        ])
        
        if dara_count > 0 and kurdelі_count > 0:
            errors.append("Cannot have both дара and күрделі set")
        
        if dara_count == 0 and kurdelі_count == 0:
            errors.append("Must have either дара or күрделі set")
        
        if dara_count > 1:
            errors.append("Only one дара type should be set")
        
        return errors
    
    def _validate_semantics(self, semantics: Dict, pos_tag: str) -> List[str]:
        """Validate semantics structure"""
        errors = []
        
        # Different POS tags have different semantic rules
        non_nan_count = sum(1 for v in semantics.values() if v != 'NaN')
        
        if non_nan_count == 0:
            errors.append("At least one semantic field must be set")
        
        return errors
    
    def fix(self, output: Dict, pos_tag: str) -> Dict:
        """Attempt to fix common issues in output"""
        
        fixed = output.copy()
        
        # Fix morphology
        if 'morphology' in fixed:
            fixed['morphology'] = self._fix_morphology(fixed['morphology'])
        
        # Fix semantics
        if 'semantics' in fixed:
            fixed['semantics'] = self._fix_semantics(fixed['semantics'], pos_tag)
        
        # Ensure word and POS tag match
        fixed['POS tag'] = pos_tag
        if 'word' not in fixed or not fixed['word']:
            fixed['word'] = fixed.get('lemma', 'unknown')
        
        return fixed
    
    def _fix_morphology(self, morphology: Dict) -> Dict:
        """Fix morphology issues"""
        
        fixed = morphology.copy()
        
        # Count дара and күрделі
        dara_negizgi = fixed.get('дара, негізгі', 'NaN')
        dara_tuyndy = fixed.get('дара, туынды', 'NaN')
        
        has_dara = (dara_negizgi != 'NaN' or dara_tuyndy != 'NaN')
        
        kurdeli_keys = [
            'күрделі, біріккен, Бірік.',
            'күрделі, қосарланған, Қос.',
            'күрделі, қысқарған, Қыс.',
            'күрделі, тіркескен, Тірк.'
        ]
        
        has_kurdeli = any(fixed.get(k, 'NaN') != 'NaN' for k in kurdeli_keys)
        
        # If both дара and күрделі, clear күрделі (дара takes precedence)
        if has_dara and has_kurdeli:
            for k in kurdeli_keys:
                fixed[k] = 'NaN'
        
        # If neither, default to дара, негізгі
        if not has_dara and not has_kurdeli:
            fixed['дара, негізгі'] = 'негізгі'
            fixed['дара, туынды'] = 'NaN'
        
        return fixed
    
    def _fix_semantics(self, semantics: Dict, pos_tag: str) -> Dict:
        """Fix semantics issues"""
        
        fixed = semantics.copy()
        
        # Ensure at least one non-NaN value
        non_nan_count = sum(1 for v in fixed.values() if v != 'NaN')
        
        if non_nan_count == 0 and fixed:
            # Set first semantic field to its key value
            first_key = list(fixed.keys())[0]
            fixed[first_key] = first_key
        
        return fixed


class MorphologyEnsemble:
    """
    Main ensemble orchestrator combining all specialized models.
    """
    
    def __init__(self, 
                 grammar_data: Dict,
                 structure_model_path: str = "Qwen/Qwen2.5-7B-Instruct",
                 lexical_model_path: str = "Qwen/Qwen2.5-14B-Instruct",
                 sozjasam_model_path: str = "microsoft/Phi-3-mini-4k-instruct",
                 pattern_db_path: Optional[str] = None):
        
        self.structure_model = StructureModel(structure_model_path)
        self.lexical_model = LexicalModel(lexical_model_path)
        self.sozjasam_model = SozjasamModel(sozjasam_model_path, pattern_db_path)
        self.validator = OutputValidator(grammar_data)
        
        self.models_loaded = False
    
    def load_models(self):
        """Load all models"""
        logger.info("Loading ensemble models...")
        
        logger.info("Loading structure model...")
        self.structure_model.load_model()
        
        logger.info("Loading lexical model...")
        self.lexical_model.load_model()
        
        logger.info("Loading sozjasam model...")
        self.sozjasam_model.load_model()
        
        self.models_loaded = True
        logger.info("All ensemble models loaded successfully")
    
    def predict(self, word: str, pos_tag: str, 
                use_validation: bool = True) -> MorphologyOutput:
        """
        Main prediction pipeline.
        
        Args:
            word: Input word
            pos_tag: POS tag
            use_validation: Whether to validate and fix output
            
        Returns:
            MorphologyOutput object
        """
        
        if not self.models_loaded:
            self.load_models()
        
        logger.info(f"Analyzing: {word} ({pos_tag})")
        
        # Stage 1: Get structure (morphology + semantics)
        logger.info("Stage 1: Structure analysis...")
        structure_output = self.structure_model.predict(word, pos_tag)
        structure_confidence = self.structure_model.get_confidence(structure_output)
        logger.info(f"Structure confidence: {structure_confidence:.2f}")
        
        # Stage 2: Get lexical meaning
        logger.info("Stage 2: Lexical meaning generation...")
        lexical_output = self.lexical_model.predict(word, pos_tag, structure_output)
        lexical_confidence = self.lexical_model.get_confidence(lexical_output)
        logger.info(f"Lexical confidence: {lexical_confidence:.2f}")
        
        # Stage 3: Get word formation pattern
        logger.info("Stage 3: Sozjasam pattern prediction...")
        sozjasam_output = self.sozjasam_model.predict(word, pos_tag, structure_output)
        sozjasam_confidence = self.sozjasam_model.get_confidence(sozjasam_output)
        logger.info(f"Sozjasam confidence: {sozjasam_confidence:.2f}")
        
        # Merge outputs
        final_output = structure_output.copy()
        final_output['lexics'] = lexical_output
        final_output['sozjasam'] = sozjasam_output
        
        # Stage 4: Validate and fix
        if use_validation:
            logger.info("Stage 4: Validation...")
            is_valid, errors = self.validator.validate(final_output, pos_tag)
            
            if not is_valid:
                logger.warning(f"Validation errors: {errors}")
                logger.info("Attempting to fix...")
                final_output = self.validator.fix(final_output, pos_tag)
                
                # Revalidate
                is_valid, errors = self.validator.validate(final_output, pos_tag)
                if is_valid:
                    logger.info("Output fixed successfully")
                else:
                    logger.warning(f"Could not fix all errors: {errors}")
        
        # Calculate overall confidence
        overall_confidence = (
            structure_confidence * 0.5 +  # Structure is most important
            lexical_confidence * 0.25 +
            sozjasam_confidence * 0.25
        )
        
        # Create output object
        result = MorphologyOutput(
            pos_tag=final_output.get('POS tag', pos_tag),
            word=final_output.get('word', word),
            lemma=final_output.get('lemma', word),
            morphology=final_output.get('morphology', {}),
            semantics=final_output.get('semantics', {}),
            lexics=final_output.get('lexics', {}),
            sozjasam=final_output.get('sozjasam', {}),
            confidence=overall_confidence,
            source="ensemble"
        )
        
        logger.info(f"Final confidence: {overall_confidence:.2f}")
        
        return result
    
    def predict_batch(self, words: List[Tuple[str, str]], 
                     batch_size: int = 8) -> List[MorphologyOutput]:
        """Predict for multiple words"""
        results = []
        
        for i in range(0, len(words), batch_size):
            batch = words[i:i+batch_size]
            for word, pos_tag in batch:
                result = self.predict(word, pos_tag)
                results.append(result)
        
        return results


def create_ensemble(grammar_data_path: str, 
                   pattern_db_path: Optional[str] = None) -> MorphologyEnsemble:
    """Factory function to create ensemble"""
    
    # Load grammar data
    with open(grammar_data_path, 'r', encoding='utf-8') as f:
        grammar_data = json.load(f)
    
    # Create ensemble
    ensemble = MorphologyEnsemble(
        grammar_data=grammar_data,
        pattern_db_path=pattern_db_path
    )
    
    return ensemble