# Personal Website Template

This repository contains a template for a personal website built using Jekyll, a static site generator. It's based on a fork of [Jekyll Now](https://github.com/barryclark/jekyll-now) and incorporates design elements inspired by [Jon Barron's website](https://jonbarron.info/).

## Features

- Simple and clean design
- Easy to customize
- Markdown support for content creation
- Thumbnail generation for images
- Multiple post categories with different formatting

## Usage

Feel free to use this template for your own personal website. However, please note that all images and content in the `images`, `pdfs`, and `_posts` folders are copyrighted and should not be reused without permission.

## Customization

1. Update `_config.yml` with your personal information
2. Replace content in `_posts` with your own posts
3. Add your own images to the `images` folder
4. Run `_make_thumbnails.sh` to generate thumbnails for your images

## Known Issues

- Multiple entries in `sitemap.xml` for `index.html` due to forced `permalink: /`
- Extra metadata in posts from previous Jekyll design (can be ignored)
- Thumbnail generation required for image sizing
- Different formatting for three post categories requires changes in multiple places when adjusting sizing

## Local Development

1. Install Jekyll and bundler gems
```
gem install jekyll bundler
```
2. Clone this repository
3. Run `bundle install`
4. Start the local server
```
bundle exec jekyll serve
```
5. View your website at `http://localhost:4000`

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgements

- [Jekyll Now](https://github.com/barryclark/jekyll-now) for the initial template
- [Jon Barron](https://jonbarron.info/) for design inspiration

For more information on Jekyll, visit the [Jekyll documentation](https://jekyllrb.com/docs/).