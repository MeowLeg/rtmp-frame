create table pic (
    id integer primary key autoincrement,
    path text not null default '',
    stream_url text not null default '',
    project_uuid text not null default '',
    organization_uuid text not null default '',
    pic_md5 text not null default '',
    create_date text default (strftime('%Y-%m-%d %H:%M:%S', datetime('now', 'localtime'))),
    predicted integer not null default 0
);

create table predict (
    id integer primary key autoincrement,
    path text not null default '',
    tag text not null default '',
    predict_path text not null default '',
    stream_url text not null default '',
    project_uuid text not null default '',
    organization_uuid text not null default '',
    pic_md5 text not null default '',
    create_date text default (strftime('%Y-%m-%d %H:%M:%S', datetime('now', 'localtime')))
);
