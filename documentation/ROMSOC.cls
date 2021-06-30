\ProvidesClass{ROMSOC}
\LoadClass[a4paper,final,11pt,headings=small]{scrartcl}
% --------------------------------------------
% Load packages
% --------------------------------------------
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{xspace}
\usepackage{url}
\usepackage{latexsym}
\usepackage{enumerate}
\usepackage{rotating}
\usepackage{framed}
\usepackage{fancybox}
\usepackage[automark,headsepline]{scrlayer-scrpage}
\usepackage{float}
\usepackage{times}
\usepackage{array}
\usepackage{booktabs} 
\usepackage{amsmath,amsthm, amsfonts, amssymb, amsxtra,amsopn}
\usepackage{setspace}
\usepackage{multirow}
\usepackage{nccmath}
\usepackage{tikz}
\usepackage[center,bf]{caption}
\usepackage{lastpage}
\usetikzlibrary{calc}
% Color Links
\usepackage[pdftex,pdfpagelabels,hyperfigures,hyperindex,bookmarksnumbered,colorlinks, bookmarks, breaklinks, linktocpage,citecolor=blue,urlcolor=blue,linkcolor=black,pdftitle={ROMSOC},pdfauthor={ROMSOC}]{hyperref}
\usepackage{graphicx}
\usepackage[a4paper,margin=2cm,heightrounded]{geometry}
\usepackage{ifthen}
\usepackage{pdfpages}

%%------------------------------------------
%%-- Define ROMSOC colors
%%------------------------------------------
\definecolor{ROMSOCred}{RGB}{255,0,137}
\definecolor{ROMSOCblue}{RGB}{63,162,255}
\definecolor{ROMSOCviolet}{RGB}{89,0,159}


\graphicspath{{images/}}
\DeclareGraphicsExtensions{.pdf,.png,.jpg}
 
%%--Set the section number depth.
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{4}
\setcounter{page}{2}


\RedeclareSectionCommands[
  beforeskip=-.75\baselineskip,
  afterskip=.5\baselineskip
]{section,subsection,subsubsection}
\RedeclareSectionCommands[
  beforeskip=.5\baselineskip,
  afterskip=-1em]{paragraph,subparagraph}


 
\bibliographystyle{IEEEtran}
\sloppy
%%--------------------------------------------
%%-- Definition and Quotation commands.
%%--------------------------------------------
\newtheorem{dbdefine}{Definition}[section]
\newtheorem{dbtheorem}{Theorem}[section]
\newtheorem{dblemma}{Lemma}[section]
\newtheorem{dbnotate}{Notation}[section]
\newtheorem{dbaxiom}{Axiom}[section]
\newenvironment{dbproof}[1][Proof]
{\noindent \textbf{#1: }}{\hfill $\blacksquare$}
 
%-----------------------------------------------
%-- Float related environments and commands.
%-----------------------------------------------
%--Modify behavior of the figure and table environment.
\def\fps@figure{hbt}
\def\fnum@figure{\textbf{\figurename~\thefigure}}
\def\fps@table{htb}
\def\fnum@table{\textbf{\tablename~\thetable}}
\def\mathfigure{\thesisfiguretrue\figure}
\def\endmathfigure{\endfigure\thesisfigurefalse}

% --------------------------------------------
% Headers
% --------------------------------------------
\pagestyle{scrheadings}
\renewcommand{\sectionmark}[1]{\markright{\thesection.\ #1}}

\makeatletter
\makeindex
\makeglossary
\setcounter{tocdepth}{4}
\setcounter{secnumdepth}{4}
\newcommand{\indic}{\mathtt{1}\!\!\mathtt{l}}
\newcommand{\proba}{\mathbb{P}}
\newcommand{\esper}{\mathbb{E}}
\newcommand{\Nats}{I\!\!N}
\newcommand{\nats}{I\!\!N}
\newcommand{\Reals}{I\!\!R}
\newcommand{\reals}{I\!\!R}
\newcommand{\espalm}{\mathbb{E}_N^{o}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% No indent and small paragraph sep.
\setlength{\parindent}{0mm}
\addtolength{\parskip}{1mm}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Floating figures
\renewcommand\floatpagefraction{.9}
\renewcommand\topfraction{.9}
\renewcommand\bottomfraction{.9}
\renewcommand\textfraction{.1}
\setcounter{totalnumber}{50}
\setcounter{topnumber}{50}
\setcounter{bottomnumber}{50}

\endinput

%%--The \tableofcontents.
\renewcommand\tableofcontents{%
  \renewcommand{\contentsname}{Table of Contents}
  \if@twocolumn
  \@restonecoltrue\onecolumn
  \else
  \@restonecolfalse
  \fi
  \oldsection*{\contentsname
    \@mkboth{\uppercase{\contentsname}}{\uppercase{\contentsname}}}%
  \@starttoc{toc}%
  \if@restonecol\twocolumn\fi}

%% --The \listoffigures command.
\renewcommand\listoffigures{%
  \if@twocolumn
  \@restonecoltrue\onecolumn
  \else
  \@restonecolfalse
  \fi
  \oldchapter*{\listfigurename
    \@mkboth{\uppercase{\listfigurename}}{\uppercase{\listfigurename}}}%
  \@starttoc{lof}%
  \if@restonecol\twocolumn\fi}

%% --The \listoftables command.
\renewcommand\listoftables{%
  \if@twocolumn%
  \@restonecoltrue\onecolumn%
  \else%
  \@restonecolfalse%
  \fi%
  \oldchapter*{\listtablename%
    \@mkboth{\uppercase{\listtablename}}{\uppercase{\listtablename}}}%
  \@starttoc{lot}%
  \if@restonecol\twocolumn\fi}

\endinput