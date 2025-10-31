create table if not exists pic (
    id integer primary key autoincrement,
    path text not null default '',
    uuid text not null default '',
    stream_url text not null default '',
    project_uuid text not null default '',
    organization_uuid text not null default '',
    stream_md5 text not null default '',
    predicted integer not null default 0,
    create_date text default (strftime('%Y-%m-%d %H:%M:%S', datetime('now', 'localtime')))
);

create table if not exists predict (
    id integer primary key autoincrement,
    path text not null default '',
    uuid text not null default '',
    tag text not null default '',
    predict_path text not null default '',
    stream_url text not null default '',
    project_uuid text not null default '',
    organization_uuid text not null default '',
    stream_md5 text not null default '',
    create_date text default (strftime('%Y-%m-%d %H:%M:%S', datetime('now', 'localtime')))
);

create table if not exists stream (
    id integer primary key autoincrement,
    uuid text not null default '',
    stream_url text not null default '',
    project_uuid text not null default '',
    organization_uuid text not null default '',
    sn text not null default '',
    is_over integer not null default 0,
    create_date text default (strftime('%Y-%m-%d %H:%M:%S', datetime('now', 'localtime')))
);

create table if not exists stream_tag (
    id integer primary key autoincrement,
    uuid text not null default '',
    code text not null default '',
    name text not null default ''
);
