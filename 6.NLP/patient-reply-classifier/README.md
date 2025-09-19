# Detect Open Responses

<!-- vscode-markdown-toc -->
* 1. [Requirements](#Requirements)
* 2. [Deploy](#Deploy)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

##  1. <a name='Requirements'></a>Requirements

You need a service account credentials 

##  2. <a name='Deploy'></a>Deploy

Deploy test:

```
make deploy-image-test
make deploy-container-test
```

Deploy prod:

```
make deploy-image
make deploy-container
```
