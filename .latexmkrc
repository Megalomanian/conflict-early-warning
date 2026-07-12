# XeLaTeX compilation for Chinese + English paper
$pdf_mode = 1;
$xelatex = "xelatex -interaction=nonstopmode -synctex=1 %O %S";
$bibtex = "bibtex %O %S";
$clean_ext = "bbl blg log aux out synctex.gz fls fdb_latexmk run.xml xdv nav snm toc vrb";
