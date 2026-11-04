"""
A imagem da gravação: o tratamento do vídeo da pessoa (celular, WhatsApp, webcam).

- **original:** como veio.
- **realce:** os níveis esticados (o preto e o branco de verdade, medidos no 1º quadro
  pela luminância, para não piscar nem tingir), o balanço de branco pelos tons quase
  neutros (a parede), um pouco mais de contraste e de cor, um
  tom levemente quente e a nitidez (máscara de desfoque) que devolve o detalhe que a
  compressão comeu. Com o realce, a gravação pequena (480p do WhatsApp) sobe até a
  ``--resolucao`` pedida.

Só a gravação passa por aqui; as fotos das cenas e o personagem ficam como estão.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

NIVEIS = ("original", "realce")

#: Os percentis que viram o preto e o branco (o resto do histograma é esticado entre eles).
_BAIXO, _ALTO = 0.6, 99.9
#: Contraste (curva em S), saturação e o calor (ganho no vermelho, corte no azul).
_CONTRASTE, _SATURACAO, _CALOR = 0.15, 1.08, 0.02
#: O quanto o balanço de branco pode mexer em cada canal (8%).
_BALANCO = 0.08


class Realce:
    """Trata quadro a quadro com os níveis medidos uma vez só."""

    def __init__(self) -> None:
        self._niveis: tuple[np.float32, np.float32] | None = None
        self._ganhos: np.ndarray | None = None

    def _medir(self, rgb: np.ndarray) -> tuple[np.float32, np.float32]:
        # pela luminância, o mesmo ganho nos três canais: esticar cada canal à parte tingia
        # a parede (o cinza-azulado virava ciano)
        luz = rgb[::4, ::4].astype(np.float32).mean(axis=2)
        baixo = float(np.percentile(luz, _BAIXO))
        alto = float(np.percentile(luz, _ALTO))
        return np.float32(baixo), np.float32(max(alto - baixo, 32.0))

    @staticmethod
    def _balanco(rgb: np.ndarray) -> np.ndarray:
        """Os ganhos por canal que deixam neutro o que é quase cinza (parede, camiseta
        branca): o cinza-azulado da lâmpada fria vira cinza. Limitados a ±_BALANCO, para
        não apagar uma parede colorida de verdade."""
        a = rgb[::4, ::4].reshape(-1, 3).astype(np.float32)
        neutros = a[((a.max(axis=1) - a.min(axis=1)) < 30) & (a.mean(axis=1) > 60)]
        if len(neutros) < 0.05 * len(a):
            return np.ones(3, np.float32)
        media = neutros.mean(axis=0)
        return np.clip(media.mean() / media, 1 - _BALANCO, 1 + _BALANCO).astype(np.float32)

    def __call__(self, matriz: np.ndarray) -> np.ndarray:
        rgb = matriz[..., :3]
        if self._niveis is None:
            self._niveis = self._medir(rgb)
            self._ganhos = self._balanco(rgb)
        baixo, faixa = self._niveis
        x = np.clip((rgb.astype(np.float32) * self._ganhos - baixo) / faixa, 0.0, 1.0)
        x = x - _CONTRASTE * x * (1 - x) * (1 - 2 * x)          # curva em S suave
        cinza = x.mean(axis=2, keepdims=True)
        x = cinza + (x - cinza) * _SATURACAO
        x[..., 0] *= 1 + _CALOR
        x[..., 2] *= 1 - _CALOR
        saida = Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))
        saida = saida.filter(ImageFilter.UnsharpMask(radius=1.6, percent=70, threshold=2))
        rgb_novo = np.asarray(saida)
        if matriz.shape[2] == 4:
            return np.dstack([rgb_novo, matriz[..., 3]])
        return rgb_novo


def tratamento(nivel: str):
    """A função que trata cada quadro, ou ``None`` no original."""
    return Realce() if nivel == "realce" else None


__all__ = ["NIVEIS", "Realce", "tratamento"]
