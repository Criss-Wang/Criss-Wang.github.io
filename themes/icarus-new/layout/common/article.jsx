const moment = require('moment');
const { Component, Fragment } = require('inferno');
const Share = require('./share');
const Donates = require('./donates');
const Comment = require('./comment');
const ArticleLicensing = require('hexo-component-inferno/lib/view/misc/article_licensing');

/**
 * Get the word count of text.
 */
function getWordCount(content) {
    if (typeof content === 'undefined') {
        return 0;
    }
    content = content.replace(/<\/?[a-z][^>]*>/gi, '');
    content = content.trim();
    return content ? (content.match(/[\u00ff-\uffff]|[a-zA-Z]+/g) || []).length : 0;
}

module.exports = class extends Component {
    render() {
        const { config, helper, page, index } = this.props;
        const { article, plugins } = config;
        const { url_for, date, date_xml, __, _p } = helper;

        const indexLaunguage = config.language || 'en';
        const language = page.lang || page.language || config.language || 'en';
        const cover = page.cover ? url_for(page.cover) : null;
        const updateTime = article && article.update_time !== undefined ? article.update_time : true;
        const isUpdated = page.updated && !moment(page.date).isSame(moment(page.updated));
        const shouldShowUpdated = page.updated && ((updateTime === 'auto' && isUpdated) || updateTime === true);
        const isProject = page.categories !== undefined && page.categories.findOne({name: "Projects"}) !== undefined;
        

        return <Fragment>
            {/* Main content */}
            <div class="card">
                {/* Thumbnail */}
                {cover ? <div class="card-image">
                    {index ? <a href={url_for(page.path)} class="image is-7by3">
                        <img class="fill" src={cover} alt={page.title || cover} />
                    </a> : <span class="image is-7by3">
                        <img class="fill" src={cover} alt={page.title || cover} />
                    </span>}
                </div> : null}
                <article class={`card-content article${'direction' in page ? ' ' + page.direction : ''}`} role="article">
                    {/* Metadata */}
                    {/* Title */}

                    {!page.title.includes("About") ? (
                        <h1 className="title is-size-4 is-size-5-mobile has-text-weight-normal">
                            {index ? (
                                <a className="has-link-black-ter" href={url_for(page.path)}>
                                    {page.title}
                                </a>
                            ) : (
                                page.title
                            )}
                        </h1>
                    ) : null}
                    {page.layout !== 'page' ? <div class="article-meta is-size-7 level is-mobile">
                        <div class="level-left">
                            {/* Creation Date */}
                            {page.date && <span class="level-item">
                                <i className="far fa-calendar-check">&nbsp;</i>
                                {isProject ? 
                                <span>
                                    <time dateTime={date_xml(page.date)} title={date_xml(page.date)}>{date(page.date)}</time>
                                    &nbsp; to &nbsp;
                                    <time dateTime={date_xml(page.updated)} title={date_xml(page.updated)}>{date(page.updated)}</time>
                                </span> 
                                : 
                                <time dateTime={date_xml(page.date)} title={date_xml(page.date)}>{date(page.date)}</time>}
                                
                            </span>}

                            {/* author */}
                            {page.author ? <span class="level-item"> {page.author} </span> : null}
                            {/* Categories */}
                            {page.categories && page.categories.length ? <span class="level-item">
                                <i class="far fa-folder-open">&nbsp;</i>
                                {(() => {
                                    const categories = [];
                                    const max_categories = 4;
                                    page.categories.forEach((category, j) => {
                                        if (category !== "Projects") {
                                            if (j > max_categories) {
                                                return;
                                            }
                                            categories.push(<a class="link-muted" href={url_for(category.path)}>{category.name}</a>);
                                            if (j < Math.min(page.categories.length - 1, max_categories)) {
                                                categories.push(<span>&nbsp;,&nbsp;</span>);
                                            }
                                        }
                                    });
                                    return categories;
                                })()}
                            </span> : null}

                            {/* Visitor counter */}
                            {!index && plugins && plugins.busuanzi === true ? <span class="level-item" id="busuanzi_container_page_pv" dangerouslySetInnerHTML={{
                                __html: _p('plugin.visit_count', '<span id="busuanzi_value_page_pv">0</span>')
                            }}></span> : null}
                        </div>
                    </div> : null}
                    {/* Title */}
                    {/* {page.title !== '' ? <h1 class="title is-3 is-size-4-mobile">
                        {index ? <a class="link-muted" href={url_for(page.link || page.path)}>{page.title}</a> : page.title}
                    </h1> : null} */}
                    {
                        index && page.link ? <div><img src={page.link} /></div> : null
                    }
                    {/* Content/Excerpt */}
                    {!page.title.includes("About") ? <div>{index && page.excerpt ? <hr style="background-color:grey;height:1px;margin:1rem 0"></hr> : <hr style="background-color:grey"></hr>}</div> : null}
                    <div style="padding-bottom:5px"></div>
                    {index && page.excerpt ? <div class="content" style="margin-bottom: 0px !important" dangerouslySetInnerHTML={{ __html: page.excerpt }}></div>
                        :
                        <div class="content" dangerouslySetInnerHTML={{ __html: page.content }}></div>}

                    {/* Licensing block */}
                    {!index && article && article.licenses && Object.keys(article.licenses)
                        ? <ArticleLicensing.Cacheable page={page} config={config} helper={helper} /> : null}
                    {index && page.excerpt ? <hr style="height:0.5px;margin:0.5rem 0" /> : <hr style="height:1px;margin:1rem 0" />}

                    {/* Share button */}
                    {/* {!index ? <Share config={config} page={page} helper={helper} /> : null} */}
                </article>
            </div>
            {/* Donate button
            {!index ? <Donates config={config} helper={helper} /> : null} */}
            {/* Post navigation */}
            {!index && (page.prev || page.next) ? <nav class="post-navigation mt-4 level is-mobile">
                {page.prev ? <div class="level-start">
                    <a class={`article-nav-prev level level-item${!page.prev ? ' is-hidden-mobile' : ''} link-muted`} href={url_for(page.prev.path)}>
                        <i class="level-item fas fa-chevron-left"></i>
                        <span class="level-item">{page.prev.title}</span>
                    </a>
                </div> : null}
                {page.next ? <div class="level-end">
                    <a class={`article-nav-next level level-item${!page.next ? ' is-hidden-mobile' : ''} link-muted`} href={url_for(page.next.path)}>
                        <span class="level-item">{page.next.title}</span>
                        <i class="level-item fas fa-chevron-right"></i>
                    </a>
                </div> : null}
            </nav> : null}
            {/* Comment */}
            {/* {!index ? <Comment config={config} page={page} helper={helper} /> : null} */}
        </Fragment>;
    }
};
