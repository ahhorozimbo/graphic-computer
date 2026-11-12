# Computação Gráfica

Trabalhos da disciplina de Computação Gráfica.

| Pasta | O que é | Tecnologia |
|---|---|---|
| [`Filtros/`](Filtros) | Editor de imagens no navegador: rotação, redimensionamento, escala de cinza, binarização e filtros de média, mediana e gaussiano implementados manualmente sobre os pixels | HTML, JavaScript, Tailwind |
| [`anim/`](anim) | Animação de formas geométricas girando em `<canvas>` | HTML, JavaScript |
| [`findplate/`](findplate) | Detecção de placas de veículos com classificador Haar Cascade | Python, OpenCV |

## Como rodar

**Filtros e anim:** abra o `index.html` da pasta no navegador.

**findplate:**

```bash
cd findplate
pip install opencv-python matplotlib
python count.py
```

Para testar outra imagem, troque o caminho em `count.py` (há três exemplos em `images/`).
