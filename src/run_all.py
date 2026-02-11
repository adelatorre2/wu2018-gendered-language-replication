from __future__ import annotations

from pathlib import Path
import shutil
import sys
import textwrap

from utils_run import ensure_output_dirs, raw_pkg_dir, repo_root, run_cmd


def copy_outputs(src_dir: Path, dest_dir: Path, patterns: list[str]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    for pattern in patterns:
        matches = sorted(src_dir.glob(pattern))
        if not matches:
            missing.append(pattern)
            continue
        for path in matches:
            shutil.copy2(path, dest_dir / path.name)
    if missing:
        raise FileNotFoundError(
            f"Expected outputs not found in {src_dir}: {', '.join(missing)}"
        )


def run_lasso_scripts(raw_dir: Path, logs_dir: Path, out_dir: Path) -> None:
    lasso_dir = raw_dir / "lasso"
    python = sys.executable

    run_cmd(
        [python, "lasso-logit-full-sample.py"],
        log_path=logs_dir / "lasso_logit_full.log",
        cwd=lasso_dir,
    )
    copy_outputs(
        raw_dir,
        out_dir / "lasso_full",
        [
            "coef_lasso_logit_full.txt",
            "ypred_train.txt",
            "ypred_test0.txt",
            "ypred_test1.txt",
            "i_keep_columns.txt",
        ],
    )

    run_cmd(
        [python, "lasso-logit-pronoun-sample.py"],
        log_path=logs_dir / "lasso_logit_pronoun.log",
        cwd=lasso_dir,
    )
    copy_outputs(
        raw_dir,
        out_dir / "lasso_pronoun",
        [
            "coef_lasso_logit_pronoun.txt",
            "ypred_pronoun_train.txt",
            "ypred_pronoun_test0.txt",
            "ypred_pronoun_test1.txt",
            "i_keep_columns.txt",
        ],
    )


def run_tables_and_figures(raw_dir: Path, logs_dir: Path, out_root: Path) -> None:
    lasso_dir = raw_dir / "lasso"

    out_fig = (out_root / "figures" / "figure1.pdf").resolve()
    out_tab1 = (out_root / "tables" / "table1.csv").resolve()
    out_tab2 = (out_root / "tables" / "table2.csv").resolve()

    r_code = textwrap.dedent(
        f"""
        out_fig <- "{out_fig.as_posix()}"
        out_tab1 <- "{out_tab1.as_posix()}"
        out_tab2 <- "{out_tab2.as_posix()}"

        source("../tables-figures.R")

        write.csv(tab1, out_tab1, row.names=FALSE)
        write.csv(tab2, out_tab2, row.names=FALSE)

        library(ggplot2)
        pdf(out_fig)
        p <- ggplot(trend,aes(x=date))+
          geom_point(aes(y=frac,color=as.factor(female),shape=as.factor(female)))+
          geom_point(aes(y=frac_pronoun,color=as.factor(female),shape=as.factor(female)))+
          geom_line(aes(y=frac,color=as.factor(female)))+
          geom_line(aes(y=frac_pronoun,color=as.factor(female)),linetype=2)+
          scale_color_manual(values=c("#56B4E9","#CC79A7"),labels=c("Male","Female"))+
          scale_shape_discrete(labels=c("Male","Female"))+
          scale_x_date(date_breaks="1 month",date_labels =  "%b %Y")+
          scale_y_continuous(breaks=seq(0.05,0.20,by=0.05),limits=c(0.05,0.20))+
          xlab("Month of the Latest Update")+ylab("Fraction of Female (Male) Posts")+
          theme(legend.position="bottom",legend.title=element_blank(),panel.background = element_blank(),
                axis.line = element_line(colour = "black"),axis.title=element_text(size=10),
                axis.text.x=element_text(angle = 90,vjust=-0.005,size=9))
        print(p)
        dev.off()
        """
    ).strip()

    run_cmd(
        ["Rscript", "-e", r_code],
        log_path=logs_dir / "tables_figures.log",
        cwd=lasso_dir,
    )


def main() -> int:
    ensure_output_dirs()

    root = repo_root()
    raw_dir = raw_pkg_dir()
    if not raw_dir.exists():
        raise FileNotFoundError(
            "Raw OpenICPSR package not found. Expected directory: "
            f"{raw_dir}"
        )
    out_root = root / "output"
    logs_dir = out_root / "logs"
    intermediate_dir = out_root / "intermediate"

    run_lasso_scripts(raw_dir, logs_dir, intermediate_dir)
    run_tables_and_figures(raw_dir, logs_dir, out_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
