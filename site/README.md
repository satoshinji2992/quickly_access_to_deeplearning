# 课程项目页

这是“快速入门深度学习”的项目页，首页集中呈现课程路线、关键配图与实现检查，文档区由 Hugo 生成。

正文仍以仓库中的章节和练习 README 为唯一来源。构建脚本会处理内部链接和配图，再交给 Hugo 输出静态页面：

```bash
python scripts/build_docs_site.py
hugo server --source site
```

浏览器打开 `http://localhost:1313/quickly_access_to_deeplearning/` 即可预览。GitHub Pages 在 `main` 更新后执行相同的构建流程。
