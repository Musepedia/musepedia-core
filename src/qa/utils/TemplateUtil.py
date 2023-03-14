import jinja2


class TemplateUtil:

    def __init__(self, template_dir_path: str):
        self._template_loader = jinja2.FileSystemLoader(searchpath=template_dir_path)
        self._template_environment = jinja2.Environment(loader=self._template_loader)

    def render_template(self, template_path: str, variable_dict: dict) -> str:
        template = self._template_environment.get_template(template_path)

        return template.render(variable_dict)


if __name__ == '__main__':
    template_util = TemplateUtil('../templates/')
    variables = {
        'qa_prompt': False,
        'exhibit_description': '银杏（学名：Ginkgo biloba），落叶乔木，寿命可达3000年以上。',
        'exhibit_label': '银杏'
    }
    print(template_util.render_template('exhibit_introduction.jinja', variables))

